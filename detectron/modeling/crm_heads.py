from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
from detectron.modeling.ResNet import add_stage

import detectron.modeling.init as param_init
import detectron.modeling.triplet_loss as triplet_loss
import detectron.utils.reid as reid_utils
'''

'''


def add_crm_outputs(model, blob_in, dim, preprefix='crm'):
    if not model.train:  # == if test
        return

    dim_inner = cfg.REID.BPM_DIM
    im_per_batch = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
    if cfg.FPN.FPN_ON and cfg.REID.FPN_SHARED:
        im_per_batch *= cfg.REID.FPN_NUM
    roi_per_im = len(blob_in)

    prefix = preprefix

    s, _ = model.net.Concat(
        blob_in, [prefix + '_feat', prefix + '_feat_concat_split_info'],
        axis=1,
        add_axis=1)

    s, _ = model.net.Reshape(
        s, [prefix + '_feat_r', prefix + '_feat_shape'],
        shape=[im_per_batch * roi_per_im, dim_inner])

    model.FC(
        prefix + '_feat_r',
        prefix + '_fc8c',
        dim_inner,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))
    model.FC(
        prefix + '_feat_r',
        prefix + '_fc8d',
        dim_inner,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))

    model.Softmax(prefix + '_fc8c', prefix + '_alpha_cls', axis=1)

    model.net.Reshape(
        prefix + '_fc8d', [prefix + '_fc8d_r', prefix + '_fc8_shape'],
        shape=[im_per_batch, roi_per_im, model.num_classes - 1])
    model.Transpose(prefix + '_fc8d_r', prefix + '_fc8d_t', axes=(0, 2, 1))
    model.Softmax(prefix + '_fc8d_t', prefix + '_alpha_det_t', axis=2)
    model.Transpose(
        prefix + '_alpha_det_t', prefix + '_alpha_det_r', axes=(0, 2, 1))
    model.net.Reshape(
        prefix + '_alpha_det_r',
        [prefix + '_alpha_det', prefix + '_alpha_det_r_shape'],
        shape=[im_per_batch * roi_per_im, model.num_classes - 1])

    model.net.Mul([prefix + '_alpha_cls', prefix + '_alpha_det'],
                  prefix + '_rois_pred')

    if not model.train and False:  # == if test
        model.net.ReduceSum(
            prefix + '_rois_pred',
            prefix + '_rois_conf',
            axes=[1],
            keepdims=False)

        model.net.Mul(
            [prefix + '_feat_r', prefix + '_rois_conf'],
            prefix + '_feat_r_s',
            broadcast=True,
            axis=0,
        )

        model.net.Reshape(
            prefix + '_feat_r_s',
            [prefix + '_feature_concat', prefix + '_r_s_shape'],
            shape=[im_per_batch, roi_per_im * dim_inner])

        if cfg.REID.NORMALIZE_FEATURE:
            feat_norm = triplet_loss.normalize(
                model,
                prefix + '_feature_concat',
                prefix + '_feature_concat_norm',
                axis=1)


def add_crm_losses(model, preprefix='crm'):
    loss_gradients = {}

    im_per_batch = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
    if cfg.FPN.FPN_ON and cfg.REID.FPN_SHARED:
        im_per_batch *= cfg.REID.FPN_NUM

    if cfg.FPN.FPN_ON and cfg.REID.FPN_SHARED:
        labels_oh = model.net.Tile(
            'labels_oh', 'labels_oh_tiled', axis=0, tiles=cfg.REID.FPN_NUM)
        labels_int32 = 'labels_int32_tiled'
    else:
        labels_oh = 'labels_oh'
        labels_int32 = 'labels_int32'

    prefix = preprefix

    model.net.Reshape(
        prefix + '_rois_pred',
        [prefix + '_rois_pred_r', prefix + '_rois_pred_shape'],
        shape=[im_per_batch, -1, model.num_classes - 1])

    model.net.ReduceSum(
        prefix + '_rois_pred_r', prefix + '_prob', axes=[1], keepdims=False)

    cross_entropy = model.net.CrossEntropyWithLogits(
        [prefix + '_prob', labels_oh], [prefix + '_cross_entropy'])

    loss_cls = model.net.AveragedLoss([cross_entropy], [prefix + '_loss'])

    loss_gradients.update(
        reid_utils.get_loss_gradients_weighted(model, [loss_cls], 1.0))
    model.Accuracy([prefix + '_prob', labels_int32], prefix + '_accuracy')
    model.AddLosses([prefix + '_loss'])
    model.AddMetrics(prefix + '_accuracy')

    return loss_gradients
