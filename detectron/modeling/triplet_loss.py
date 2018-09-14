from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
import detectron.utils.blob as blob_utils
import detectron.modeling.init as param_init
import detectron.utils.reid as reid_utils


def normalize(model, x, y, axis=-1, prefix=''):
    z = model.net.Normalize(x, y, axis=axis)
    return z


def euclidean_dist(model, x, y, m, n, prefix=''):
    model.net.ConstantFill([], [prefix + '_m'], shape=[1], value=m)
    model.net.ConstantFill([], [prefix + '_n'], shape=[1], value=n)
    model.net.Concat(
        [prefix + '_m', prefix + '_n'],
        [prefix + '_mn', prefix + '_mn_concat_split_info'],
        axis=0)
    model.net.Concat(
        [prefix + '_n', prefix + '_m'],
        [prefix + '_nm', prefix + '_nm_concat_split_info'],
        axis=0)

    x1 = model.net.Copy(x, prefix + '_x1')
    x2 = model.net.Copy(x, prefix + '_x2')
    y1 = model.net.Copy(y, prefix + '_y1')
    y2 = model.net.Copy(y, prefix + '_y2')

    model.net.Pow(x1, prefix + '_xx', exponent=2.0)
    model.net.Pow(y1, prefix + '_yy', exponent=2.0)

    model.net.ReduceSum(
        prefix + '_xx', prefix + '_xx_sum', axes=[1], keepdims=True)
    model.net.ReduceSum(
        prefix + '_yy', prefix + '_yy_sum', axes=[1], keepdims=True)

    model.net.Expand([prefix + '_xx_sum', prefix + '_mn'],
                     [prefix + '_xx_exp'])
    model.net.Expand([prefix + '_yy_sum', prefix + '_nm'],
                     [prefix + '_yy_exp'])
    model.Transpose(prefix + '_yy_exp', prefix + '_yy_exp_t', axes=(1, 0))

    model.net.Add([prefix + '_xx_exp', prefix + '_yy_exp_t'],
                  [prefix + '_xx_yy'])

    model.net.MatMul([x2, y2], [prefix + '_xy'], trans_b=1)
    model.net.Scale(prefix + '_xy', prefix + '_2xy', scale=2.0)

    model.net.Sub([prefix + '_xx_yy', prefix + '_2xy'], [prefix + '_xy_dist'])

    model.net.Clip(prefix + '_xy_dist', prefix + '_xy_dist', min=1e-12)

    model.net.Sqrt(prefix + '_xy_dist', prefix + '_xy_dist_sqrt')
    return prefix + '_xy_dist_sqrt'


def euclidean_dist2(model, x, n, prefix=''):
    model.net.ConstantFill([], [prefix + '_n'], shape=[1], value=n)
    model.net.Concat(
        [prefix + '_n', prefix + '_n'],
        [prefix + '_nn', prefix + '_nm_concat_split_info'],
        axis=0)

    model.net.Pow(x, prefix + '_xx', exponent=2.0)

    model.net.ReduceSum(
        prefix + '_xx', prefix + '_xx_sum', axes=[1], keepdims=True)

    model.net.Expand([prefix + '_xx_sum', prefix + '_nn'],
                     [prefix + '_xx_exp'])
    model.Transpose(prefix + '_xx_exp', prefix + '_xx_exp_t', axes=(1, 0))

    model.net.Add([prefix + '_xx_exp', prefix + '_xx_exp_t'],
                  [prefix + '_xx_yy'])

    model.net.MatMul([x, x], [prefix + '_xy'], trans_b=1)
    model.net.Scale(prefix + '_xy', prefix + '_2xy', scale=2.0)

    model.net.Sub([prefix + '_xx_yy', prefix + '_2xy'], [prefix + '_xy_dist'])
    return prefix + '_xy_dist'


def hard_example_mining(model, dist, labels, N, return_inds=False, prefix=''):
    model.net.ConstantFill([], [prefix + '_N'], shape=[1], value=N)
    model.net.Concat(
        [prefix + '_N', prefix + '_N'],
        [prefix + '_NN', prefix + '_NN_concat_split_info'],
        axis=0)

    model.net.Expand([labels, prefix + '_NN'], [prefix + '_labels_exp'])
    model.Transpose(
        prefix + '_labels_exp', prefix + '_labels_exp_t', axes=(1, 0))

    model.net.EQ([prefix + '_labels_exp', prefix + '_labels_exp_t'],
                 [prefix + '_is_pos'])
    # model.net.NE([prefix + '_labels_exp', prefix + '_labels_exp_t'],
    # [prefix + '_is_neg'])

    model.net.StopGradient(prefix + '_is_pos', prefix + '_is_pos')

    model.net.Cast(
        prefix + '_is_pos', prefix + '_is_pos_c', to=1, from_type='bool')

    model.net.Mul([dist, prefix + '_is_pos_c'], prefix + '_dist_pos')

    model.net.Scale(prefix + '_is_pos_c', prefix + '_is_pos_c_s', scale=1e16)
    model.net.Add([dist, prefix + '_is_pos_c_s'], prefix + '_dist_neg')

    model.net.ReduceMax(
        prefix + '_dist_pos', prefix + '_dist_ap', axes=[1], keepdims=True)
    model.net.ReduceMin(
        prefix + '_dist_neg', prefix + '_dist_an', axes=[1], keepdims=True)

    return prefix + '_dist_ap', prefix + '_dist_an'


def add_triplet_losses(model,
                       blob_in,
                       labels,
                       N,
                       loss_weight=1.0,
                       margin=0.2,
                       prefix='tri'):
    if cfg.REID.NORMALIZE_FEATURE:
        feat_norm = normalize(model, blob_in, prefix + '_norm', axis=1)
        model.net.Squeeze(feat_norm, prefix + '_x', dims=[2, 3])
        # model.net.Squeeze(feat_norm, prefix + '_y', dims=[2, 3])
    else:
        model.net.Squeeze(blob_in, prefix + '_x', dims=[2, 3])
        # model.net.Squeeze(blob_in, prefix + '_y', dims=[2, 3])

    # dist = euclidean_dist(
    # model, prefix + '_x', prefix + '_y', N, N, prefix=prefix)

    model.net.PairWiseDistance(prefix + '_x', prefix + '_dist')
    model.net.Clip(prefix + '_dist', prefix + '_dist_clip', min=1e-12)
    dist = model.net.Sqrt(prefix + '_dist_clip', prefix + '_dist_sqrt')

    # dist_ap, dist_an = hard_example_mining(
    # model, dist, labels, N, prefix=prefix)

    dist_ap, dist_an = model.net.BatchHard(
        [dist, labels], [prefix + '_dist_ap', prefix + '_dist_an'])

    Y = model.net.ConstantFill(
        [], [prefix + '_Y'], shape=[N, 1], value=-1, dtype=2)
    mrc = model.net.MarginRankingCriterion(
        [dist_ap, dist_an, Y], [prefix + '_mrc'], margin=margin)

    if cfg.REID.TRIPLET_LOSS_CROSS:
        mrc_ave = model.net.ReduceMean(
            [mrc], [prefix + '_ave_mrc'], axes=[0], keepdims=False)
        model.AddMetrics(prefix + '_ave_mrc')

        loss_scale_factor = model.param_init_net.ConstantFill(
            [], 'loss_scale_factor', shape=[1], value=0.0)

        loss_triplet = model.net.Mul(
            [mrc_ave, loss_scale_factor],
            prefix + '_triplet_loss',
            broadcast=True)
    else:
        loss_triplet = model.net.AveragedLoss([mrc],
                                              [prefix + '_triplet_loss'])

    loss_gradients = reid_utils.get_loss_gradients_weighted(
        model, [loss_triplet], loss_weight)
    model.AddLosses([prefix + '_triplet_loss'])

    model.net.ReduceMean(
        prefix + '_dist_ap',
        prefix + '_dist_ap_mean',
        axes=[0],
        keepdims=False)
    model.net.ReduceMean(
        prefix + '_dist_an',
        prefix + '_dist_an_mean',
        axes=[0],
        keepdims=False)
    model.AddMetrics(prefix + '_dist_ap_mean')
    model.AddMetrics(prefix + '_dist_an_mean')

    return loss_gradients
