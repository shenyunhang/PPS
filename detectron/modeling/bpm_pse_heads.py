from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
import detectron.modeling.init as param_init

from detectron.modeling.ResNet import add_stage
'''
Beyond Part Models Person Retrieval with Refined Part Pooling (and a Strong Convolutional Baseline)
A Pose-Sensitive Embedding for Person Re-Identification with Expanded Cross Neighborhood Re-Ranking
'''


def add_attr_outputs(model, blob_in, dim):
    # attr predictor stream
    num_attr = cfg.REID.PSE_VIEW
    prefix = 'attr'

    dim_inner = 256
    i = 0
    current = model.Conv(
        blob_in,
        prefix + '_conv' + str(i),
        dim,
        dim_inner,
        3,
        stride=2,
        pad=1,
        weight_init=('MSRAFill', {}),
        bias_init=('ConstantFill', {
            'value': 0.
        }),
        # weight_init=param_init.kaiming_uniform_(
        # [256, dim, 1, 1], a=math.sqrt(5)),
        # bias_init=param_init.bias_init_([256, dim, 1, 1]),
        no_bias=0)
    current = model.SpatialBN(
        current, prefix + '_bn' + str(i), dim_inner, is_test=not model.train)
    current = model.Relu(current, current)
    for i in range(1, 2):
        current = model.Conv(
            current,
            prefix + '_conv' + str(i),
            dim_inner,
            dim_inner,
            3,
            stride=2,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            # weight_init=param_init.kaiming_uniform_(
            # [256, dim, 1, 1], a=math.sqrt(5)),
            # bias_init=param_init.bias_init_([256, dim, 1, 1]),
            no_bias=0)
        current = model.SpatialBN(
            current,
            prefix + '_bn' + str(i),
            dim_inner,
            is_test=not model.train)
        current = model.Relu(current, current)

    current = model.AveragePool(current, prefix + '_pool', global_pooling=True)

    current = model.FC(
        current,
        prefix + '_fc',
        dim_inner,
        num_attr,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0))

    current = model.Softmax(current, prefix + '_softmax')
    return current


def add_bpm_pse_outputs2(model, blob_in, dim):
    current = add_attr_outputs(model, blob_in, dim)

    # ReID stream
    num_attr = cfg.REID.PSE_VIEW
    attr_pred_list = ['attr_pred_' + str(i) for i in range(num_attr)]
    model.net.Split(
        current, attr_pred_list, split=[1 for i in range(num_attr)], axis=1)

    attr_feature_list = []
    dim_out = int(2048 / cfg.REID.PSE_VIEW)
    dim_inner = int(dim_out / 8)
    for attr_id in range(num_attr):
        # prefix = '_[v{}]'.format(attr_id)
        prefix = 'v{}'.format(attr_id)
        shape_0 = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
        model.net.Reshape(
            attr_pred_list[attr_id],
            [attr_pred_list[attr_id], 'attr_old_shape_' + str(attr_id)],
            shape=[shape_0, 1, 1, 1])

        current = model.Conv(
            blob_in,
            prefix + '_conv1',
            dim,
            dim_out,
            3,
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            # weight_init=param_init.kaiming_uniform_(
            # [256, dim, 1, 1], a=math.sqrt(5)),
            # bias_init=param_init.bias_init_([256, dim, 1, 1]),
            no_bias=0)
        current = model.SpatialBN(
            current, prefix + '_bn1', dim_out, is_test=not model.train)
        current = model.Relu(current, current)

        current = model.net.Mul(
            [current, attr_pred_list[attr_id]],
            prefix + '_scale',
            broadcast=True,
        )
        attr_feature_list.append(current)

    current = model.net.Sum(
        [attr_feature for attr_feature in attr_feature_list], 'v_scale')

    add_bpm_outputs(model, current, dim_out)


def add_bpm_pse_outputs(model, blob_in, dim):
    current = add_attr_outputs(model, blob_in, dim)

    # ReID stream
    num_attr = cfg.REID.PSE_VIEW
    attr_pred_list = ['attr_pred_' + str(i) for i in range(num_attr)]
    model.net.Split(
        current, attr_pred_list, split=[1 for i in range(num_attr)], axis=1)

    attr_feature_list = []
    dim_out = int(2048 / cfg.REID.PSE_VIEW)
    dim_inner = int(dim_out / 2)
    for attr_id in range(num_attr):
        # prefix = '_[v{}]'.format(attr_id)
        prefix = 'v{}'.format(attr_id)
        shape_0 = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
        model.net.Reshape(
            attr_pred_list[attr_id],
            [attr_pred_list[attr_id], 'attr_old_shape_' + str(attr_id)],
            shape=[shape_0, 1, 1, 1])

        current = model.Conv(
            blob_in,
            prefix + '_conv1',
            dim,
            dim_inner,
            1,
            stride=1,
            pad=0,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            # weight_init=param_init.kaiming_uniform_(
            # [256, dim, 1, 1], a=math.sqrt(5)),
            # bias_init=param_init.bias_init_([256, dim, 1, 1]),
            no_bias=0)
        current = model.SpatialBN(
            current, prefix + '_bn1', dim_inner, is_test=not model.train)
        current = model.Relu(current, current)

        current = model.Conv(
            current,
            prefix + '_conv2',
            dim_inner,
            dim_inner,
            3,
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            # weight_init=param_init.kaiming_uniform_(
            # [256, dim, 1, 1], a=math.sqrt(5)),
            # bias_init=param_init.bias_init_([256, dim, 1, 1]),
            no_bias=0)
        current = model.SpatialBN(
            current, prefix + '_bn2', dim_inner, is_test=not model.train)
        current = model.Relu(current, current)

        current = model.Conv(
            current,
            prefix + '_conv3',
            dim_inner,
            dim_out,
            1,
            stride=1,
            pad=0,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            # weight_init=param_init.kaiming_uniform_(
            # [256, dim, 1, 1], a=math.sqrt(5)),
            # bias_init=param_init.bias_init_([256, dim, 1, 1]),
            no_bias=0)
        current = model.SpatialBN(
            current, prefix + '_bn3', dim_out, is_test=not model.train)
        current = model.Relu(current, current)

        current = model.net.Mul(
            [current, attr_pred_list[attr_id]],
            prefix + '_scale',
            broadcast=True,
        )
        attr_feature_list.append(current)

    current = model.net.Sum(
        [attr_feature for attr_feature in attr_feature_list], 'v_scale')

    blob_out, dim_out = add_ResNet_split_head(model, blob_in, dim_out, 16)
    add_bpm_outputs(model, blob_out, dim_out)


def add_bpm_pse_outputs_c4(model, blob_in, dim):
    current = add_attr_outputs(model, blob_in, dim)

    # ReID stream
    num_attr = cfg.REID.PSE_VIEW
    attr_pred_list = ['attr_pred_' + str(i) for i in range(num_attr)]
    model.net.Split(
        current, attr_pred_list, split=[1 for i in range(num_attr)], axis=1)

    attr_feature_list = []
    for attr_id in range(num_attr):
        prefix = '_[v{}]'.format(attr_id)
        shape_0 = cfg.TRAIN.IMS_PER_BATCH if model.train else 1
        model.net.Reshape(
            attr_pred_list[attr_id],
            [attr_pred_list[attr_id], 'attr_old_shape_' + str(attr_id)],
            shape=[shape_0, 1, 1, 1])

        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        current, dim_out = add_stage(
            model,
            prefix + '_res5',
            blob_in,
            3,
            dim,
            2048,
            dim_bottleneck * 8,
            cfg.RESNETS.RES5_DILATION,
            stride_init=cfg.RESNETS.RES5_STRIDE)

        current = model.net.Mul(
            [current, attr_pred_list[attr_id]],
            prefix + '_scale',
            broadcast=True,
        )
        attr_feature_list.append(current)

    current = model.net.Sum(
        [attr_feature for attr_feature in attr_feature_list], 'v_scale')

    add_bpm_outputs(model, current, dim_out)


def add_bpm_pse_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    # View prediction loss
    attr_cls_prob, attr_loss_cls = model.net.SoftmaxWithLoss(
        ['attr_fc', 'attr_labels_int32', 'attr_weight'],
        ['attr_cls_prob', 'attr_loss_cls'],
        scale=model.GetLossScale())

    model.Accuracy(['attr_cls_prob', 'attr_labels_int32'], 'attr_accuracy_cls')
    model.AddMetrics('attr_accuracy_cls')
    loss_gradients = get_loss_gradients(
        model, [attr_loss_cls], weight=cfg.REID.PSE_WEIGHT)
    model.AddLosses([attr_loss_cls])

    loss_gradients.update(add_bpm_losses(model))
    return loss_gradients


def get_loss_gradients(model, loss_blobs, weight):
    """Generate a gradient of 1 for each loss specified in 'loss_blobs'"""
    loss_gradients = {}
    for b in loss_blobs:
        loss_grad = model.net.ConstantFill(
            b, [b + '_grad'], value=1.0 * weight)
        loss_gradients[str(b)] = str(loss_grad)
    return loss_gradients
