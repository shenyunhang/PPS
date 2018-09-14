from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import logging
import numpy as np
import os
import pprint
import yaml

from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import load_cfg
from detectron.utils.io import load_object
from detectron.utils.io import save_object
import detectron.utils.c2 as c2_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_loss_scale(model, scale):
    name = 'loss_scale_factor'
    for gpu_id in range(cfg.NUM_GPUS):
        with c2_utils.CudaScope(gpu_id):
            blob_name = 'gpu_' + str(gpu_id) + '/' + name
            ws_blob = workspace.FetchBlob(blob_name)
            if ws_blob[0] != scale:
                workspace.FeedBlob(blob_name,
                                   np.array([scale], dtype=np.float32))

                print('Set {} to {}'.format(name, scale))


def get_loss_gradients_weighted(model, loss_blobs, loss_weight):
    """Generate a gradient of 1 for each loss specified in 'loss_blobs'"""
    loss_gradients = {}
    for b in loss_blobs:
        loss_grad = model.net.ConstantFill(
            b, [b + '_grad'], value=1.0 * loss_weight)
        loss_gradients[str(b)] = str(loss_grad)
    return loss_gradients


def get_loss_gradients_weighted_scaled(model, loss_blobs, loss_weight):
    """Generate a gradient of 1 for each loss specified in 'loss_blobs'"""
    loss_gradients = {}
    for b in loss_blobs:
        for i, key in enumerate(cfg.REID.LOSS_KEYS):
            if key in str(b):
                loss_scale = cfg.REID.LOSS_SCALE_NAMES[i]
                model.param_init_net.ConstantFill(
                    [], loss_scale, shape=[1], value=0.0)

        loss_grad = model.net.ConstantFill(
            b, [b + '_grad'], value=1.0 * loss_weight)
        loss_grad = model.net.Mul(
            [loss_grad, loss_scale],
            b + '_grad',
            broadcast=True,
            axis=0,
        )
        loss_gradients[str(b)] = str(loss_grad)
    return loss_gradients


def get_loss_gradients_weighted_(model, loss_blobs, loss_weight):
    if len(cfg.REID.LOSS_KEYS) > 0:
        get_loss_gradients_weighted_scaled(model, loss_blobs, loss_weight)
    """Generate a gradient of 1 for each loss specified in 'loss_blobs'"""
    loss_gradients = {}
    for b in loss_blobs:
        loss_scale = model.param_init_net.ConstantFill(
            [], [b + '_scale'], shape=[1], value=1.0 * loss_weight)
        loss_shape = model.net.Shape(b, b + '_shape')
        loss_grad = model.net.Expand([loss_scale, loss_shape], [b + '_grad'])
        loss_gradients[str(b)] = str(loss_grad)
    return loss_gradients
