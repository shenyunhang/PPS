from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
from detectron.modeling.ResNet import add_stage

import detectron.modeling.init as param_init
import detectron.modeling.bpm_heads as bpm_heads
import detectron.utils.reid as reid_utils


def add_apm_outputs2(model, blob_in, dim):
    bpm_heads.add_bpm_outputs(model, blob_in, dim)

    prefix_list = ['feature_' + str(i) for i in range(cfg.REID.BPM_STRIP_NUM)]
    dim_inner = cfg.REID.BPM_DIM
    im_per_batch = cfg.TRAIN.IMS_PER_BATCH if model.train else 1

    fc_list = [prefix + '_fc' for prefix in prefix_list]
    model.net.Concat(fc_list, ['fc8c', 'fc8c_split_info'], add_axis=1, axis=1)

    feature_list = [prefix + '_conv' for prefix in prefix_list]
    model.net.Concat(
        feature_list, ['fc7', 'fc7_split_info'], add_axis=1, axis=1)
    model.net.Reshape(
        'fc7', ['fc7_', 'fc7_shape'],
        shape=[im_per_batch * cfg.REID.BPM_STRIP_NUM, dim_inner])

    model.FC(
        'fc7_',
        'fc8d_',
        dim_inner,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))

    model.net.Reshape(
        'fc8d_', ['fc8d', 'fc8d__shape'],
        shape=[im_per_batch, cfg.REID.BPM_STRIP_NUM, model.num_classes - 1])

    model.Softmax('fc8c', 'alpha_cls', axis=2)
    model.Transpose('fc8d', 'fc8d_t', axes=(0, 2, 1))
    model.Softmax('fc8d_t', 'alpha_det_t', axis=2)
    model.Transpose('alpha_det_t', 'alpha_det', axes=(0, 2, 1))
    model.net.Mul(['alpha_cls', 'alpha_det'], 'rois_pred')


def add_apm_outputs1(model, blob_in, dim):
    # Box classification layer
    model.FC(
        blob_in,
        'fc8c',
        dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))
    model.FC(
        blob_in,
        'fc8d',
        dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))

    model.Softmax('fc8c', 'alpha_cls', axis=1)
    model.Transpose('fc8d', 'fc8d_t', axes=(1, 0))
    model.Softmax('fc8d_t', 'alpha_det_t', axis=1)
    model.Transpose('alpha_det_t', 'alpha_det', axes=(1, 0))
    model.net.Mul(['alpha_cls', 'alpha_det'], 'rois_pred')

    if not model.train:  # == if test
        # model.net.Alias('rois_pred', 'cls_prob')
        model.net.Mul(['fc8c', 'fc8d'], 'fc8cd')

        add_cls_pred('fc8cd', 'raw_cls_prob', model)
        add_cls_pred('rois_pred', 'cls_prob', model)

        model.net.ReduceSum('rois_pred', 'rois_conf', axes=[1], keepdims=False)

        model.net.Mul(
            [blob_in, 'rois_conf'],
            blob_in + '_scale',
            broadcast=True,
            axis=0,
        )

        model.net.ReduceSum(
            blob_in + '_scale', 'feature_concat', axes=[0], keepdims=False)


def add_apm_outputs3(model, blob_in, dim):
    bpm_heads.add_bpm_outputs(model, blob_in[0], dim[0])
    blob_in = blob_in[1]
    dim = dim[1]

    prefix = 'global'
    dim_inner = cfg.REID.BPM_DIM

    # model.StopGradient(blob_in, blob_in)

    model.Conv(
        blob_in,
        prefix + '_conv',
        dim,
        dim_inner,
        1,
        stride=1,
        pad=0,
        weight_init=('MSRAFill', {}),
        bias_init=('ConstantFill', {
            'value': 0.
        }),
        no_bias=0)
    model.Relu(prefix + '_conv', prefix + '_conv')

    model.FC(
        prefix + '_conv',
        # blob_in,
        'fc8c',
        dim_inner,
        # dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))
    model.FC(
        prefix + '_conv',
        # blob_in,
        'fc8d',
        dim_inner,
        # dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))

    model.Softmax('fc8c', 'alpha_cls', axis=1)

    im_per_batch = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
    strip_num = cfg.REID.BPM_STRIP_NUM
    strip_len = cfg.REID.BPM_STRIP_LEN

    # roi_per_im = cfg.TRAIN.BATCH_SIZE_PER_IM
    roi_per_im = strip_num * strip_len * int(strip_num * strip_len / 3)

    model.net.Reshape(
        'fc8d', ['fc8d_r', 'fc8_shape'],
        shape=[im_per_batch, roi_per_im, model.num_classes - 1])
    model.Transpose('fc8d_r', 'fc8d_t', axes=(0, 2, 1))
    model.Softmax('fc8d_t', 'alpha_det_t', axis=2)
    model.Transpose('alpha_det_t', 'alpha_det_r', axes=(0, 2, 1))
    model.net.Reshape(
        'alpha_det_r', ['alpha_det', 'alpha_det_r_shape'],
        shape=[im_per_batch * roi_per_im, model.num_classes - 1])

    model.net.Mul(['alpha_cls', 'alpha_det'], 'rois_pred')

    if not model.train:  # == if test
        # model.net.Alias('rois_pred', 'cls_prob')
        model.net.Mul(['fc8c', 'fc8d'], 'fc8cd')

        # add_cls_pred('fc8cd', 'raw_cls_prob', model)
        # add_cls_pred('rois_pred', 'cls_prob', model)

        model.net.ReduceSum('rois_pred', 'rois_conf', axes=[1], keepdims=False)

        model.net.Mul(
            [prefix + '_conv', 'rois_conf'],
            # [blob_in, 'rois_conf'],
            prefix + '_scale',
            broadcast=True,
            axis=0,
        )

        model.net.Reshape(
            prefix + '_scale', [prefix + '_scale_r', prefix + '_scale_shape'],
            shape=[im_per_batch, roi_per_im, dim_inner])
        # shape=[im_per_batch, roi_per_im, dim])

        model.net.ReduceMean(
            prefix + '_scale_r',
            prefix + '_feature_concat',
            axes=[1],
            keepdims=False)


def add_apm_outputs(model, blob_in, dim, spatial_scale, preprefix='apm'):
    bpm_heads.add_bpm_outputs(model, blob_in[0], dim[0], preprefix='bpm')

    blob_in = blob_in[1]
    dim = dim[1]

    dim_inner = cfg.REID.BPM_DIM

    im_per_batch = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
    strip_num = cfg.REID.BPM_STRIP_NUM
    strip_len = int(cfg.REID.SCALE[1] * spatial_scale / strip_num)
    roi_per_im = int(cfg.REID.SCALE[1] * spatial_scale * cfg.REID.SCALE[0] *
                     spatial_scale / strip_num)

    feature_list = []
    scale_feature_list = []

    for i in range(strip_num):
        prefix = preprefix + str(i)

        # s=model.StopGradient(blob_in[i], prefix + '_' + blob_in[i])
        # s = model.StopGradient(blob_in[i], blob_in[i])
        s = blob_in[i]

        model.Conv(
            s,
            prefix + '_conv',
            dim,
            dim_inner,
            1,
            stride=1,
            pad=0,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            no_bias=0)
        model.Relu(prefix + '_conv', prefix + '_conv')

        model.FC(
            prefix + '_conv',
            prefix + '_fc8c',
            dim_inner,
            model.num_classes - 1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0))
        model.FC(
            prefix + '_conv',
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

        if not model.train:  # == if test
            model.net.Mul([prefix + '_fc8c', prefix + '_fc8d'],
                          prefix + '_fc8cd')

            model.net.Reshape(
                prefix + '_fc8cd',
                [prefix + '_fc8cd_r', prefix + '_fc8cd_shape'],
                shape=[im_per_batch, roi_per_im * (model.num_classes - 1)])

            model.net.Reshape(
                prefix + '_conv', [prefix + '_conv_r', prefix + '_conv_shape'],
                shape=[im_per_batch, roi_per_im, dim_inner])

            model.net.ReduceMean(
                prefix + '_conv_r',
                prefix + '_feature',
                axes=[1],
                keepdims=False)

            feature_list.append(prefix + '_feature')

        if True:
            model.net.ReduceSum(
                prefix + '_rois_pred',
                prefix + '_rois_conf',
                axes=[1],
                keepdims=False)

            model.net.Mul(
                # [prefix + '_conv', prefix + '_rois_conf'],
                [blob_in[i], prefix + '_rois_conf'],
                prefix + '_scale',
                broadcast=True,
                axis=0,
            )

            model.net.Reshape(
                prefix + '_scale',
                [prefix + '_scale_r', prefix + '_scale_shape'],
                # shape=[im_per_batch, roi_per_im, dim_inner, 1, 1])
                shape=[im_per_batch, roi_per_im, dim, 1, 1])

            model.net.ReduceMean(
                prefix + '_scale_r',
                prefix + '_scale_feature',
                axes=[1],
                keepdims=False)

            scale_feature_list.append(prefix + '_scale_feature')

    if not model.train:  # == if test
        prefix = preprefix
        model.net.Concat(
            feature_list, [
                preprefix + '_feature_concat',
                preprefix + '_feature_concat_split_info'
            ],
            axis=1)

        model.net.Concat(
            scale_feature_list, [
                preprefix + '_scale_feature_concat',
                preprefix + '_scale_feature_concat_split_info'
            ],
            axis=1)

    # bpm_heads.add_bpm_outputs(model, scale_feature_list, dim_inner, preprefix='bpm')
    bpm_heads.add_bpm_outputs(model, scale_feature_list, dim, preprefix='abpm')


def add_cls_pred(in_blob, out_blob, model):
    if cfg.TRAIN.IMS_PER_BATCH == 1:
        model.net.ReduceSum(in_blob, out_blob, axes=[0], keepdims=True)
        return

    model.net.RoIScoreReshape(
        [in_blob, 'rois'],
        in_blob + '_reshape',
        num_classes=model.num_classes - 1,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        rois_size=cfg.TRAIN.BATCH_SIZE_PER_IM)
    model.net.RoIScorePool(
        in_blob + '_reshape', out_blob, num_classes=model.num_classes - 1)


def add_apm_losses1(model):
    add_cls_pred('rois_pred', 'cls_prob', model)

    cls_prob_softmax, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_prob', 'labels_int32'], ['cls_prob_softmax', 'loss_cls'],
        scale=model.GetLossScale())

    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls])
    model.Accuracy(['cls_prob_softmax', 'labels_int32'], 'accuracy_cls')
    model.AddLosses(['loss_cls'])
    model.AddMetrics('accuracy_cls')

    return loss_gradients


def add_apm_losses2(model):
    prefix = 'global'
    model.net.ReduceSum(
        'rois_pred', prefix + '_cls_prob', axes=[1], keepdims=False)

    cross_entropy = model.net.CrossEntropyWithLogits(
        [prefix + '_cls_prob', 'labels_oh'], [prefix + '_cross_entropy'])

    loss_cls = model.net.AveragedLoss([cross_entropy], [prefix + '_loss_cls'])

    loss_gradients = reid_utils.get_loss_gradients_weighted(
        model, [loss_cls], 1.0)
    model.Accuracy([prefix + '_cls_prob', 'labels_int32'],
                   prefix + '_accuracy_cls')
    model.AddLosses([prefix + '_loss_cls'])
    model.AddMetrics(prefix + '_accuracy_cls')

    # loss_gradients.update(bpm_heads.add_bpm_losses(model))

    return loss_gradients


def add_apm_losses3(model):
    loss_gradients = bpm_heads.add_bpm_losses(model)

    prefix = 'global'

    im_per_batch = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
    strip_num = cfg.REID.BPM_STRIP_NUM
    strip_len = cfg.REID.BPM_STRIP_LEN

    # roi_per_im = cfg.TRAIN.BATCH_SIZE_PER_IM
    roi_per_im = strip_num * strip_len * int(strip_num * strip_len / 3)

    model.net.Reshape(
        'rois_pred', ['rois_pred_r', 'rois_pred_shape'],
        shape=[im_per_batch, roi_per_im, model.num_classes - 1])

    model.net.ReduceSum(
        'rois_pred_r', prefix + '_prob', axes=[1], keepdims=False)

    cross_entropy = model.net.CrossEntropyWithLogits(
        [prefix + '_prob', 'labels_oh'], [prefix + '_cross_entropy'])

    loss_cls = model.net.AveragedLoss([cross_entropy], [prefix + '_loss'])

    loss_gradients.update(
        reid_utils.get_loss_gradients_weighted(model, [loss_cls], 0.1))
    model.Accuracy([prefix + '_prob', 'labels_int32'], prefix + '_accuracy')
    model.AddLosses([prefix + '_loss'])
    model.AddMetrics(prefix + '_accuracy')

    return loss_gradients


def add_apm_losses(model, spatial_scale, preprefix='apm'):
    loss_gradients = {}
    loss_gradients.update(bpm_heads.add_bpm_losses(model, preprefix='bpm'))
    loss_gradients.update(bpm_heads.add_bpm_losses(model, preprefix='abpm'))

    im_per_batch = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
    strip_num = cfg.REID.BPM_STRIP_NUM
    strip_len = int(cfg.REID.SCALE[1] * spatial_scale / strip_num)
    roi_per_im = int(cfg.REID.SCALE[1] * spatial_scale * cfg.REID.SCALE[0] *
                     spatial_scale / strip_num)

    for i in range(strip_num):
        prefix = preprefix + str(i)

        model.net.Reshape(
            prefix + '_rois_pred',
            [prefix + '_rois_pred_r', prefix + '_rois_pred_shape'],
            shape=[im_per_batch, roi_per_im, model.num_classes - 1])

        model.net.ReduceSum(
            prefix + '_rois_pred_r',
            prefix + '_prob',
            axes=[1],
            keepdims=False)

        cross_entropy = model.net.CrossEntropyWithLogits(
            [prefix + '_prob', 'labels_oh'], [prefix + '_cross_entropy'])

        loss_cls = model.net.AveragedLoss([cross_entropy], [prefix + '_loss'])

        loss_gradients.update(
            reid_utils.get_loss_gradients_weighted(model, [loss_cls], 0.01))
        model.Accuracy([prefix + '_prob', 'labels_int32'],
                       prefix + '_accuracy')
        model.AddLosses([prefix + '_loss'])
        model.AddMetrics(prefix + '_accuracy')

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #


def add_ResNet_roi_split_head(model,
                              blob_in,
                              dim_in,
                              spatial_scale,
                              preprefix='apm'):
    return bpm_heads.add_ResNet_roi_split_head(
        model, blob_in, dim_in, spatial_scale, preprefix='bpm')


def add_ResNet_roi_deep_split_head(model,
                                   blob_in,
                                   dim_in,
                                   spatial_scale,
                                   preprefix='apm'):
    im_per_batch = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
    strip_num = cfg.REID.BPM_STRIP_NUM
    strip_len = int(cfg.REID.SCALE[1] * spatial_scale / strip_num)

    model.Transpose(blob_in, blob_in + '_t', axes=(0, 2, 3, 1))

    feat_roi, _ = model.net.Reshape(
        blob_in + '_t', ['roi_feat', blob_in + '_shape'],
        shape=[
            im_per_batch * strip_num * strip_len * int(
                strip_len * strip_num / 3), dim_in, 1, 1
        ])

    feat_strip, dim_strip = bpm_heads.add_ResNet_roi_split_head(
        model, blob_in, dim_in, spatial_scale, preprefix='bpm')

    return [feat_strip, feat_roi], [dim_strip, dim_in]


def add_ResNet_roi_more_deep_split_head(model,
                                        blob_in,
                                        dim_in,
                                        spatial_scale,
                                        preprefix='apm'):
    im_per_batch = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
    strip_num = cfg.REID.BPM_STRIP_NUM
    strip_len = int(cfg.REID.SCALE[1] * spatial_scale / strip_num)

    feats_strip, dim_strip = bpm_heads.add_ResNet_roi_split_head(
        model, blob_in, dim_in, spatial_scale, preprefix='bpm')

    feats_roi = [preprefix + str(i) + '_roi_feat' for i in range(strip_num)]
    for i in range(strip_num):
        model.Transpose(
            feats_strip[i], feats_strip[i] + '_t', axes=(0, 2, 3, 1))
        feat_roi, _ = model.net.Reshape(
            feats_strip[i] + '_t', [feats_roi[i], feats_strip[i] + '_shape'],
            shape=[
                im_per_batch * strip_len * int(strip_len * strip_num / 3),
                dim_strip, 1, 1
            ])

    return [feats_strip, feats_roi], [dim_strip, dim_strip]


def add_ResNet_roi_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    roi_feat = model.net.RoIFeatureBoost([roi_feat, 'obn_scores'], roi_feat)

    roi_feat_pool = model.AveragePool(
        roi_feat, 'roi_feat_pool', global_pooling=True)

    blob_strip, dim_strip = add_ResNet_roi_split_head(model, blob_in, dim_in,
                                                      spatial_scale)

    return [blob_strip, roi_feat_pool], [dim_strip, dim_in]


def add_ResNet_roi_1fc_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    roi_feat_pool = model.AveragePool(
        roi_feat, 'roi_feat_pool', global_pooling=True)

    model.FC(roi_feat_pool, 'fc6', dim_in, hidden_dim)
    model.Relu('fc6', 'fc6')
    l = DropoutIfTraining(model, 'fc6', 'drop6', 0.5)
    return l, hidden_dim


def add_ResNet_roi_2fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    roi_feat_pool = model.AveragePool(roi_feat, 'roi_feat_pool', kernel=7)

    # roi_feat_boost = model.net.RoIFeatureBoost([roi_feat_pool, 'obn_scores'],
    # 'roi_feat_boost')

    # model.FC(roi_feat_boost, 'fc6', dim_in, hidden_dim)
    model.FC(roi_feat_pool, 'fc6', dim_in, hidden_dim)
    model.Relu('fc6', 'fc6')
    l = DropoutIfTraining(model, 'fc6', 'drop6', 0.5)
    model.FC(l, 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    l = DropoutIfTraining(model, 'fc7', 'drop7', 0.5)
    return l, hidden_dim


def add_roi_Xconv_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    # roi_feat_boost = model.net.RoIFeatureBoost([roi_feat, 'obn_scores'],
    # 'roi_feat_boost')

    # current = roi_feat_boost
    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'head_conv' + str(i + 1),
            dim_in,
            hidden_dim,
            3,
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    # current = model.AveragePool(current, 'head_pool', kernel=roi_size)

    return current, dim_in


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    # roi_feat_boost = model.net.RoIFeatureBoost([roi_feat, 'obn_scores'],
    # 'roi_feat_boost')

    # current = roi_feat_boost
    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'head_conv' + str(i + 1),
            dim_in,
            hidden_dim,
            3,
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current,
            'head_conv' + str(i + 1),
            dim_in,
            hidden_dim,
            3,
            group_gn=get_group_gn(hidden_dim),
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_ResNet_roi_conv5_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    l = model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient(l, l)

    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage(model, 'res5', 'pool5', 3, dim_in, 2048,
                          dim_bottleneck * 8, 1, stride_init)
    s = model.AveragePool(s, 'res5_pool', kernel=7)
    return s, 2048


def DropoutIfTraining(model, blob_in, blob_out, dropout_rate):
    """Add dropout to blob_in if the model is in training mode and
    dropout_rate is > 0."""
    if model.train and dropout_rate > 0:
        blob_out = model.Dropout(
            blob_in, blob_out, ratio=dropout_rate, is_test=False)
        return blob_out
    else:
        return blob_in
