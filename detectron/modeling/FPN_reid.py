# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Functions for using a Feature Pyramid Network (FPN)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np

from detectron.core.config import cfg
from detectron.modeling.generate_anchors import generate_anchors
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

# Lowest and highest pyramid levels in the backbone network. For FPN, we assume
# that all networks have 5 spatial reductions, each by a factor of 2. Level 1
# would correspond to the input image, hence it does not make sense to use it.
LOWEST_BACKBONE_LVL = 2   # E.g., "conv2"-like level
HIGHEST_BACKBONE_LVL = 5  # E.g., "conv5"-like level


# ---------------------------------------------------------------------------- #
# FPN with ResNet
# ---------------------------------------------------------------------------- #

def add_fpn_ResNet50_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet50_conv5_body, fpn_level_info_ResNet50_conv5
    )


def add_fpn_ResNet50_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(
        model,
        ResNet.add_ResNet50_conv5_body,
        fpn_level_info_ResNet50_conv5,
        P2only=True
    )


def add_fpn_ResNet101_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet101_conv5_body, fpn_level_info_ResNet101_conv5
    )


def add_fpn_ResNet101_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(
        model,
        ResNet.add_ResNet101_conv5_body,
        fpn_level_info_ResNet101_conv5,
        P2only=True
    )


def add_fpn_ResNet152_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet152_conv5_body, fpn_level_info_ResNet152_conv5
    )


def add_fpn_ResNet152_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(
        model,
        ResNet.add_ResNet152_conv5_body,
        fpn_level_info_ResNet152_conv5,
        P2only=True
    )


# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #

def add_fpn_onto_conv_body(
    model, conv_body_func, fpn_level_info_func, P2only=False
):
    """Add the specified conv body to the model and then add FPN levels to it.
    """
    # Note: blobs_conv is in revsersed order: [fpn5, fpn4, fpn3, fpn2]
    # similarly for dims_conv: [2048, 1024, 512, 256]
    # similarly for spatial_scales_fpn: [1/32, 1/16, 1/8, 1/4]

    conv_body_func(model)
    blobs_fpn, dim_fpn, spatial_scales_fpn = add_fpn(
        model, fpn_level_info_func()
    )

    if P2only:
        # use only the finest level
        return blobs_fpn[-1], dim_fpn, spatial_scales_fpn[-1]
    else:
        # use all levels
        return blobs_fpn, dim_fpn, spatial_scales_fpn


def add_fpn(model, fpn_level_info):
    """Add FPN connections based on the model described in the FPN paper."""
    # FPN levels are built starting from the highest/coarest level of the
    # backbone (usually "conv5"). First we build down, recursively constructing
    # lower/finer resolution FPN levels. Then we build up, constructing levels
    # that are even higher/coarser than the starting level.
    fpn_dim = cfg.FPN.DIM
    min_level, max_level = get_min_max_levels()
    # Count the number of backbone stages that we will generate FPN levels for
    # starting from the coarest backbone stage (usually the "conv5"-like level)
    # E.g., if the backbone level info defines stages 4 stages: "conv5",
    # "conv4", ... "conv2" and min_level=2, then we end up with 4 - (2 - 2) = 4
    # backbone stages to add FPN to.
    num_backbone_stages = (
        len(fpn_level_info.blobs) - (min_level - LOWEST_BACKBONE_LVL)
    )

    lateral_input_blobs = fpn_level_info.blobs[:num_backbone_stages]
    output_blobs = [
        'fpn_inner_{}'.format(s)
        for s in fpn_level_info.blobs[:num_backbone_stages]
    ]
    fpn_dim_lateral = fpn_level_info.dims
    xavier_fill = ('XavierFill', {})

    # For the coarsest backbone level: 1x1 conv only seeds recursion
    if fpn_dim_lateral[0] == fpn_dim:
        output_blobs[0] = lateral_input_blobs[0]
    elif cfg.FPN.USE_GN:
        # use GroupNorm
        c = model.ConvGN(
            lateral_input_blobs[0],
            output_blobs[0],  # note: this is a prefix
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            group_gn=get_group_gn(fpn_dim),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
        output_blobs[0] = c  # rename it
    else:
        model.Conv(
            lateral_input_blobs[0],
            output_blobs[0],
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
        c = model.SpatialBN(output_blobs[0], output_blobs[0] + '_bn', fpn_dim, is_test=not model.train)
        c = model.Relu(c, c)
        output_blobs[0] = c # rename it

    #
    # Step 1: recursively build down starting from the coarsest backbone level
    #

    # For other levels add top-down and lateral connections
    for i in range(num_backbone_stages - 1):
        add_topdown_lateral_module(
            model,
            output_blobs[i],             # top-down blob
            lateral_input_blobs[i + 1],  # lateral blob
            output_blobs[i + 1],         # next output blob
            # fpn_dim_lateral[i],          # top-down dimension
            fpn_dim,                     # top-down dimension
            # fpn_dim_lateral[i] if i == 0 else fpn_dim,          # top-down dimension
            fpn_dim_lateral[i + 1],      # lateral input dimension
            # fpn_dim_lateral[i + 1],      # output dimension
            fpn_dim,                     # output dimension
        )

    # Post-hoc scale-specific 3x3 convs
    blobs_fpn = []
    dims_fpn = []
    spatial_scales = []
    for i in range(num_backbone_stages):
        blobs_fpn += [output_blobs[i]]
        dims_fpn += [fpn_dim]
        spatial_scales += [fpn_level_info.spatial_scales[i]]
        continue
        if cfg.FPN.USE_GN:
            # use GroupNorm
            fpn_blob = model.ConvGN(
                output_blobs[i],
                'fpn_{}'.format(fpn_level_info.blobs[i]),
                dim_in=fpn_dim,
                dim_out=fpn_dim,
                group_gn=get_group_gn(fpn_dim),
                kernel=3,
                pad=1,
                stride=1,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
        else:
            fpn_blob = model.Conv(
                output_blobs[i],
                'fpn_{}'.format(fpn_level_info.blobs[i]),
                dim_in=fpn_dim,
                dim_out=fpn_dim,
                kernel=3,
                pad=1,
                stride=1,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
        blobs_fpn += [fpn_blob]
        dims_fpn += [fpn_dim]
        spatial_scales += [fpn_level_info.spatial_scales[i]]

    #
    # Step 2: build up starting from the coarsest backbone level
    #

    # Check if we need the P6 feature map
    if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == HIGHEST_BACKBONE_LVL + 1:
        # Original FPN P6 level implementation from our CVPR'17 FPN paper
        P6_blob_in = blobs_fpn[0]
        P6_name = P6_blob_in + '_subsampled_2x'
        # Use max pooling to simulate stride 2 subsampling
        P6_blob = model.MaxPool(P6_blob_in, P6_name, kernel=1, pad=0, stride=2)
        blobs_fpn.insert(0, P6_blob)
        dims_fpn.insert(0, fpn_dim)
        spatial_scales.insert(0, spatial_scales[0] * 0.5)

    # Coarser FPN levels introduced for RetinaNet
    if cfg.FPN.EXTRA_CONV_LEVELS and max_level > HIGHEST_BACKBONE_LVL:
        fpn_blob = fpn_level_info.blobs[0]
        dim_in = fpn_level_info.dims[0]
        for i in range(HIGHEST_BACKBONE_LVL + 1, max_level + 1):
            fpn_blob_in = fpn_blob
            if i > HIGHEST_BACKBONE_LVL + 1:
                fpn_blob_in = model.Relu(fpn_blob, fpn_blob + '_relu')
            fpn_blob = model.Conv(
                fpn_blob_in,
                'fpn_' + str(i),
                dim_in=dim_in,
                dim_out=fpn_dim,
                kernel=3,
                pad=1,
                stride=2,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
            dim_in = fpn_dim
            blobs_fpn.insert(0, fpn_blob)
            dims_fpn.insert(0, fpn_dim)
            spatial_scales.insert(0, spatial_scales[0] * 0.5)

    return blobs_fpn, dims_fpn, spatial_scales


def add_topdown_lateral_module2(
    model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral, dim_bottom
):
    lat = fpn_lateral
    td = fpn_top

    # Top-down 2x upsampling
    if 'res5_2_sum' not in str(fpn_top) or cfg.RESNETS.RES5_STRIDE == 2:
        td = model.net.UpsampleNearest(td, fpn_bottom + '_topdown', scale=2)

    # Sum lateral and top-down
    model.net.Concat([lat, td], [fpn_bottom, fpn_bottom + '_concat_split_info'], axis=1)


def add_topdown_lateral_module(
    model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral, dim_bottom
):
    """Add a top-down lateral module."""
    # model.net.Copy(fpn_lateral, fpn_bottom)
    # return

    assert dim_top == dim_bottom or dim_lateral == dim_bottom
    # Lateral 1x1 conv
    if dim_lateral == dim_bottom:
        lat = fpn_lateral
    else:
        lat = model.Conv(
            fpn_lateral,
            fpn_bottom + '_lateral',
            dim_in=dim_lateral,
            dim_out=dim_bottom,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(
                const_fill(0.0)
                if cfg.FPN.ZERO_INIT_LATERAL else ('XavierFill', {})
            ),
            bias_init=const_fill(0.0)
        )
        lat = model.SpatialBN(lat, lat + '_bn', dim_bottom, is_test=not model.train)
        lat = model.Relu(lat, lat)

    # Top-down
    if dim_top == dim_bottom:
        td = fpn_top
    else:
        td = model.Conv(
            fpn_top,
            fpn_bottom + '_lateral',
            dim_in=dim_top,
            dim_out=dim_bottom,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(
                const_fill(0.0)
                if cfg.FPN.ZERO_INIT_LATERAL else ('XavierFill', {})
            ),
            bias_init=const_fill(0.0)
            )
        td = model.SpatialBN(td, fpn_bottom + '_lateral_bn', dim_bottom, is_test=not model.train)
        td = model.Relu(td, td)

    # Top-down 2x upsampling
    if 'res5_2_sum' not in str(fpn_top) or cfg.RESNETS.RES5_STRIDE == 2:
        td = model.net.UpsampleNearest(td, fpn_bottom + '_topdown', scale=2)

    # Sum lateral and top-down
    model.net.Sum([lat, td], fpn_bottom)


def add_topdown_lateral_module_old(
    model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral
):
    """Add a top-down lateral module."""
    td = model.Conv(
        fpn_top,
        fpn_bottom + '_lateral',
        dim_in=dim_top,
        dim_out=dim_lateral,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(
            const_fill(0.0)
            if cfg.FPN.ZERO_INIT_LATERAL else ('XavierFill', {})
        ),
        bias_init=const_fill(0.0)
        )
    td = model.SpatialBN(td, fpn_bottom + '_lateral_bn', dim_lateral, is_test=not model.train)
    td = model.Relu(td, td)

    if 'res5_2_sum' not in str(fpn_top) or cfg.RESNETS.RES5_STRIDE == 2:
        td = model.net.UpsampleNearest(td, fpn_bottom + '_topdown', scale=2)
    # Sum lateral and top-down
    model.net.Sum([fpn_lateral, td], fpn_bottom)


def get_min_max_levels():
    """The min and max FPN levels required for supporting RPN and/or RoI
    transform operations on multiple FPN levels.
    """
    min_level = LOWEST_BACKBONE_LVL
    max_level = HIGHEST_BACKBONE_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level


# ---------------------------------------------------------------------------- #
# FPN level info for stages 5, 4, 3, 2 for select models (more can be added)
# ---------------------------------------------------------------------------- #

FpnLevelInfo = collections.namedtuple(
    'FpnLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)


def fpn_level_info_ResNet50_conv5():
    if cfg.RESNETS.RES5_STRIDE == 1:
        if cfg.REID.FPN_NUM == 4:
            return FpnLevelInfo(
                blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
                dims=(2048, 1024, 512, 256),
                spatial_scales=(1. / 16., 1. / 16., 1. / 8., 1. / 4.)
            )
        elif cfg.REID.FPN_NUM == 3:
            return FpnLevelInfo(
                blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum'),
                dims=(2048, 1024, 512, 256),
                spatial_scales=(1. / 16., 1. / 16., 1. / 8.)
            )
        elif cfg.REID.FPN_NUM == 2:
            return FpnLevelInfo(
                blobs=('res5_2_sum', 'res4_5_sum'),
                dims=(2048, 1024),
                spatial_scales=(1. / 16., 1. / 16.)
            )

    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet101_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet152_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )
