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
Beyond Part Models Person Retrieval with Refined Part Pooling (and a Strong Convolutional Baseline)

'''


def add_uniform_partition(model,
                          blob_in,
                          dim_in,
                          spatial_scale,
                          preprefix='bpm'):
    strip_num = cfg.REID.BPM_STRIP_NUM

    if strip_num == 7 and cfg.REID.SCALE[1] == 16 * 24:
        split = [3, 3, 4, 4, 4, 3, 3]
        scale = 16 * spatial_scale
        split = [int(s * scale) for s in split]
    elif strip_num == 5 and cfg.REID.SCALE[1] == 16 * 24:
        split = [5, 5, 4, 5, 5]
        scale = 16 * spatial_scale
        split = [int(s * scale) for s in split]
    elif strip_num == 9 and cfg.REID.SCALE[1] == 16 * 24:
        split = [2, 3, 3, 3, 3, 3, 3, 2, 2]
        scale = 16 * spatial_scale
        split = [int(s * scale) for s in split]
    elif strip_num == 10 and cfg.REID.SCALE[1] == 16 * 24:
        split = [2, 2, 2, 3, 3, 3, 3, 2, 2, 2]
        scale = 16 * spatial_scale
        split = [int(s * scale) for s in split]
    else:
        strip_h = int(cfg.REID.SCALE[1] * spatial_scale / strip_num)
        split = [strip_h for i in range(strip_num)]
    blobs_strip = [preprefix + str(i) + '_strip' for i in range(strip_num)]
    model.net.Split(blob_in, blobs_strip, split=split, axis=2)

    for i in range(strip_num):
        prefix = preprefix + str(i)
        current = blobs_strip[i]
        if cfg.REID.MAX_AVE_FEATURE:
            model.AveragePool(
                current, prefix + '_ave_pool', global_pooling=True)
            model.MaxPool(current, prefix + '_max_pool', global_pooling=True)
        else:
            model.AveragePool(current, prefix + '_pool', global_pooling=True)


def add_uniform_part_head_(model,
                           blob_in,
                           dim_in,
                           spatial_scale,
                           preprefix='bpm'):
    add_uniform_partition(
        model, blob_in, dim_in, spatial_scale, preprefix=preprefix)
    strip_num = cfg.REID.BPM_STRIP_NUM

    blobs_out = []
    dims_out = []
    for i in range(strip_num):
        prefix = preprefix + str(i)
        if cfg.REID.MAX_AVE_FEATURE:
            current = model.net.Add(
                [prefix + '_ave_pool', prefix + '_max_pool'], prefix + '_pool')
        else:
            current = prefix + '_pool'
        blobs_out.append(current)
        dims_out.append(dim_in)

    return blobs_out, dims_out


def add_uniform_part_head(model,
                          blob_in,
                          dim_in,
                          spatial_scale,
                          preprefix='bpm'):
    if not cfg.FPN.FPN_ON:
        return add_uniform_part_head_(
            model, blob_in, dim_in, spatial_scale, preprefix=preprefix)

    blobs_outs = []
    dims_outs = []
    for i in range(len(blob_in)):
        blobs_out, dims_out = add_uniform_part_head_(
            model,
            blob_in[i],
            dim_in[i],
            spatial_scale[i],
            preprefix=preprefix + str(i))
        blobs_outs.extend(blobs_out)
        dims_outs.extend(dims_out)

    return blobs_outs, dims_outs
