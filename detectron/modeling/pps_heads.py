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
import detectron.modeling.bpm_heads as bpm_heads
'''
Enumerating k-combinations

'''

pyramid_combs = [[0], [1], [2], [3], [4], [5], [0, 1], [1, 2], [2, 3], [3, 4],
                 [4, 5], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
                 [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 1, 2, 3, 4],
                 [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]


def get_prefix(s):
    if len(str(s).split('/')) >= 2:
        prefix = str(s).split('/')[1]
    else:
        prefix = s

    prefix = str(prefix).split('_')[0] + str(prefix).split('_')[2]
    return prefix


def add_pps_part_head_(model, blob_in, dim_in, spatial_scale, preprefix='pps'):
    bpm_heads.add_uniform_partition(
        model, blob_in, dim_in, spatial_scale, preprefix=preprefix)

    strip_num = cfg.REID.BPM_STRIP_NUM
    total_c = 1 << strip_num

    blobs_out = []
    dims_out = []
    for i in range(1, total_c):
        comb = []
        for j in range(strip_num):
            tmp = i
            if tmp & (1 << j):
                comb.append(j)

        #if comb not in pyramid_combs:
        #    continue

        print(comb)
        if cfg.REID.MAX_AVE_FEATURE:
            ave_blobs = [preprefix + str(c) + '_ave_pool' for c in comb]
            max_blobs = [preprefix + str(c) + '_max_pool' for c in comb]

            prefix = preprefix
            for c in comb:
                prefix += '' + str(c)

            ave_pool = model.net.Mean(ave_blobs, prefix + '_ave_pool2')
            max_pool = model.net.Max(max_blobs, prefix + '_max_pool2')
            current = model.net.Add([ave_pool, max_pool], prefix + '_pool2')
        else:
            max_blobs = [preprefix + str(c) + '_pool' for c in comb]

            prefix = preprefix
            for c in comb:
                prefix += '' + str(c)

            current = model.net.Max(max_blobs, prefix + '_pool2')
        blobs_out.append(current)
        dims_out.append(dim_in)

    return blobs_out, dims_out


def add_pps_part_head(model, blob_in, dim_in, spatial_scale, preprefix='pps'):
    if not cfg.FPN.FPN_ON:
        return add_pps_part_head_(
            model, blob_in, dim_in, spatial_scale, preprefix=preprefix)

    if not model.train:
        return add_pps_part_head_(
            model,
            blob_in[0],
            dim_in[0],
            spatial_scale[0],
            # preprefix=preprefix + '0',
            preprefix=preprefix,
        )

    # if not model.train:
    # return add_pps_part_head_(
    # model,
    # blob_in[-1],
    # dim_in[-1],
    # spatial_scale[-1],
    # preprefix=preprefix + '0',)

    blobs_outs = []
    dims_outs = []
    for i in range(len(blob_in)):
        blobs_out, dims_out = add_pps_part_head_(
            model,
            blob_in[i],
            dim_in[i],
            spatial_scale[i],
            preprefix=preprefix + '_' + str(i) + '_',
        )
        blobs_outs.extend(blobs_out)
        dims_outs.extend(dims_out)

    if not cfg.REID.FPN_SHARED:
        return blobs_outs, dims_outs

    blobs_out = []
    dims_out = []
    num_each = int(len(blobs_outs) / len(blob_in))
    for i in range(num_each):
        prefix = get_prefix(blobs_outs[i])
        print(prefix)
        blobs = [blobs_outs[j] for j in range(i, len(blobs_outs), num_each)]
        channels = sum(
            [dims_outs[j] for j in range(i, len(blobs_outs), num_each)])

        concat, _ = model.net.Concat(
            blobs, [prefix + '_concat', prefix + '_concat_split_info'], axis=0)
        blobs_out.append(concat)
        dims_out.append(dims_outs[i])

        # concat, _ = model.net.Concat(
        # blobs, [prefix + '_concat', prefix + '_concat_split_info'], axis=1)
        # blobs_out.append(concat)
        # dims_out.append(channels)

    return blobs_out, dims_out
