from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.c2 import UnscopeName
import detectron.utils.blob as blob_utils
import detectron.modeling.init as param_init
import detectron.utils.reid as reid_utils
import detectron.modeling.crm_heads as crm_heads
import detectron.modeling.triplet_loss as triplet_loss

feature_list = []
fc_list = []


def get_prefix(s):
    prefix = UnscopeName(str(s))
    # if len(str(s).split('/')) >= 2:
    # prefix = str(s).split('/')[1]
    # else:
    # prefix = s

    prefix = str(prefix).split('_')[0]
    return prefix


def add_reid_outputs(model, blob_in, dim, preprefix='reid'):
    dim_inner = cfg.REID.BPM_DIM

    for i in range(len(blob_in)):
        prefix = get_prefix(blob_in[i])
        current = blob_in[i]

        if not cfg.MODEL.USE_GN:
            current = model.Conv(
                current,
                prefix + '_conv',
                dim[i],
                dim_inner,
                1,
                stride=1,
                pad=0,
                weight_init=('MSRAFill', {}),
                bias_init=('ConstantFill', {
                    'value': 0.
                }),
                # weight_init=param_init.kaiming_uniform_(
                # [dim_inner, dim[i], 1, 1], a=math.sqrt(5)),
                # bias_init=param_init.bias_init_([dim_inner, dim[i], 1, 1]),
                no_bias=0)
            current = model.SpatialBN(
                current, prefix + '_bn', dim_inner, is_test=not model.train)
        else:
            current = model.ConvGN(
                current,
                prefix + '_conv',
                dim[i],
                dim_inner,
                1,
                group_gn=get_group_gn(dim_inner),
                stride=1,
                pad=0,
                weight_init=('MSRAFill', {}),
                bias_init=('ConstantFill', {
                    'value': 0.
                }),
                no_conv_bias=0)

        current = model.Relu(current, current)

        if UnscopeName(str(current)) not in feature_list:
            feature_list.append(UnscopeName(str(current)))

        if cfg.REID.DROPOUT_FEATURE:
            current = model.DropoutIfTraining(current, 0.2)

        current = model.FC(
            current,
            prefix + '_fc',
            dim_inner,
            model.num_classes - 1,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0))

        if UnscopeName(str(current)) not in fc_list:
            fc_list.append(UnscopeName(str(current)))

    if not model.train:  # == if test
        model.net.Concat(
            feature_list, [
                preprefix + '_feature_concat_r',
                preprefix + '_feature_concat_r_split_info'
            ],
            axis=1)

        model.net.Concat(
            fc_list, [
                preprefix + '_fc_concat_r',
                preprefix + '_fc_concat_r_split_info'
            ],
            axis=1)

        model.net.Reshape(
            preprefix + '_feature_concat_r', [
                preprefix + '_feature_concat',
                preprefix + '_feature_concat_r_shape'
            ],
            shape=[1, -1])

        model.net.Reshape(
            preprefix + '_fc_concat_r',
            [preprefix + '_fc_concat', preprefix + '_fc_concat_r_shape'],
            shape=[1, -1])

        if cfg.REID.NORMALIZE_FEATURE:
            feat_norm = triplet_loss.normalize(
                model,
                preprefix + '_feature_concat',
                preprefix + '_feature_concat_norm',
                axis=1)

    if cfg.REID.CRM:
        crm_heads.add_crm_outputs(
            model, feature_list, dim_inner, preprefix='crm')


def add_reid_head(model, blob_in, dim_in, spatial_scale, preprefix='reid'):
    blob_out = model.AveragePool(blob_in, 'reid_feature', global_pooling=True)
    return blob_out, dim_in


def add_reid_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    cls_probs = []
    loss_clss = []

    N = cfg.TRAIN.IMS_PER_BATCH
    if cfg.FPN.FPN_ON and cfg.REID.FPN_SHARED:
        labels_int32 = model.net.Tile(
            'labels_int32', 'labels_int32_tiled', axis=0, tiles=cfg.REID.FPN_NUM)
        N *= cfg.REID.FPN_NUM
    else:
        labels_int32 = 'labels_int32'

    for i in range(len(fc_list)):
        prefix = get_prefix(fc_list[i])
        fc = fc_list[i]
        cls_prob, loss_cls = model.net.SoftmaxWithLoss(
            [fc, labels_int32], [prefix + '_prob', prefix + '_loss'],
            scale=model.GetLossScale())

        cls_probs.append(cls_prob)
        loss_clss.append(loss_cls)

        model.Accuracy([prefix + '_prob', labels_int32], prefix + '_accuracy')
        model.AddMetrics(prefix + '_accuracy')
    loss_gradients = reid_utils.get_loss_gradients_weighted(
        model, loss_clss, 1.0)
    model.AddLosses(loss_clss)

    if cfg.REID.CRM:
        lg = crm_heads.add_crm_losses(model, preprefix='crm')
        loss_gradients.update(lg)

    if cfg.REID.TRIPLET_LOSS is False:
        return loss_gradients

    for i in range(len(feature_list)):
        prefix = get_prefix(fc_list[i])
        blob_in = feature_list[i]
        lg = triplet_loss.add_triplet_losses(
            model,
            blob_in,
            labels_int32,
            N,
            loss_weight=0.14,
            margin=1.4,
            prefix=prefix)
        loss_gradients.update(lg)

    return loss_gradients
