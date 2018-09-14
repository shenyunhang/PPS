from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
import detectron.modeling.init as param_init
'''
A Coarse-to-fine Pyramidal Model for Person Re-identification via Multi-Loss Dynamic Training

'''


def add_youtu_part_head_(model,
                         blob_in,
                         dim_in,
                         spatial_scale,
                         preprefix='youtu'):
    strip_num = cfg.REID.BPM_STRIP_NUM

    strip_h = int(cfg.REID.SCALE[1] * spatial_scale / strip_num)
    strip_w = int(cfg.REID.SCALE[0] * spatial_scale)

    blobs_out = []
    dims_out = []
    for strip_num in range(cfg.REID.BPM_STRIP_NUM, 0, -1):
        stride = cfg.REID.BPM_STRIP_NUM - strip_num + 1

        ave_pool = model.AveragePool(
            blob_in,
            preprefix + str(strip_num) + '_ave',
            kernel_h=stride * strip_h,
            kernel_w=strip_w,
            stride_h=strip_h,
            stride_w=1)

        max_pool = model.MaxPool(
            blob_in,
            preprefix + str(strip_num) + '_max',
            kernel_h=stride * strip_h,
            kernel_w=strip_w,
            stride_h=strip_h,
            stride_w=1)

        current = model.net.Add([ave_pool, max_pool],
                                preprefix + str(strip_num))

        blob_out = [
            preprefix + str(strip_num) + str(i) + '_strip'
            for i in range(strip_num)
        ]
        dim_out = [dim_in for i in range(strip_num)]

        model.net.Split(
            preprefix + str(strip_num),
            blob_out,
            split=[1 for i in range(strip_num)],
            axis=2)

        blobs_out.extend(blob_out)
        dims_out.extend(dim_out)

    return blobs_out, dims_out


def add_youtu_part_head(model,
                        blob_in,
                        dim_in,
                        spatial_scale,
                        preprefix='youtu'):
    if not cfg.FPN.FPN_ON:
        return add_youtu_part_head_(
            model, blob_in, dim_in, spatial_scale, preprefix=preprefix)

    blobs_outs = []
    dims_outs = []
    for i in range(len(blob_in)):
        blobs_out, dims_out = add_youtu_part_head(
            model,
            blob_in[i],
            dim_in[i],
            spatial_scale[i],
            preprefix=preprefix + str(i))
        blobs_out.append(blob_out)
        dims_outs.extend(dims_out)

    return blobs_outs, dims_outs
