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
from detectron.modeling.bpm_heads import add_bpm_outputs
from detectron.modeling.bpm_heads import add_bpm_losses


def add_bpm_attr_outputs(model, blob_in, dim):
    prefix = 'attr'
    current = model.AveragePool(blob_in, 'attr_pool', global_pooling=True)

    current = model.FC(
        current,
        'attr_fc',
        dim,
        cfg.REID.PSE_VIEW,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0))

    add_bpm_outputs(model, blob_in, dim)


def add_bpm_attr_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    # View prediction loss
    attr_cls_prob, attr_loss_cls = model.net.SoftmaxWithLoss(
        ['attr_fc', 'attr_labels_int32', 'attr_weight'],
        ['attr_cls_prob', 'attr_loss_cls'],
        scale=model.GetLossScale())

    model.Accuracy(['attr_cls_prob', 'attr_labels_int32'], 'attr_accuracy_cls')
    model.AddMetrics('attr_accuracy_cls')
    loss_gradients = blob_utils.get_loss_gradients(model, [attr_loss_cls])
    model.AddLosses([attr_loss_cls])

    loss_gradients.update(add_bpm_losses(model))
    return loss_gradients
