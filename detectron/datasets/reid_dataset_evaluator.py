from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import numpy as np
import os
import uuid
import time
from shutil import copyfile
import cv2
from collections import defaultdict
from contextlib import contextmanager
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist

from pycocotools.cocoeval import COCOeval

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.utils.io import save_object
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)


def evaluate(json_dataset, all_feats, output_dir):
    to_re_rank = cfg.REID.RERANK
    pool_type = 'average'
    verbose = True

    # Use standard Market1501 CMC settings for all datasets here.
    separate_camera_set = False
    single_gallery_shot = False
    first_match_break = True

    roidb = json_dataset.get_roidb(gt=True)

    ids, cams, im_names, marks, im_paths = [], [], [], [], []
    for entry in roidb:
        pid, cam, im_name, mark, im_path = get_info(entry)
        ids.append(pid)
        cams.append(cam)
        im_names.append(im_name)
        marks.append(mark)
        im_paths.append(im_path)

    feat = all_feats
    ids = np.hstack(ids)
    cams = np.hstack(cams)
    im_names = np.hstack(im_names)
    marks = np.hstack(marks)
    im_paths = np.hstack(im_paths)

    print('-' * 40)
    print('Starting eval')

    # with measure_time('Extracting feature...', verbose=verbose):
    # feat, ids, cams, im_names, marks = self.extract_feat(
    # normalize_feat, verbose)

    # query, gallery, multi-query indices
    q_inds = marks == 0
    g_inds = marks == 1
    mq_inds = marks == 2

    # A helper function just for avoiding code duplication.
    def compute_score(dist_mat,
                      query_ids=ids[q_inds],
                      gallery_ids=ids[g_inds],
                      query_cams=cams[q_inds],
                      gallery_cams=cams[g_inds]):
        # Compute mean AP
        mAP = mean_ap(
            distmat=dist_mat,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            query_cams=query_cams,
            gallery_cams=gallery_cams)
        # Compute CMC scores
        cmc_scores = cmc(
            distmat=dist_mat,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            query_cams=query_cams,
            gallery_cams=gallery_cams,
            separate_camera_set=separate_camera_set,
            single_gallery_shot=single_gallery_shot,
            first_match_break=first_match_break,
            topk=10)
        return mAP, cmc_scores

    def print_scores(mAP, cmc_scores):
        print(
            '[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
            .format(mAP, *cmc_scores[[0, 4, 9]]))

    ################
    # Single Query #
    ################

    with measure_time('Computing distance...', verbose=verbose):
        # query-gallery distance
        q_g_dist = compute_dist(feat[q_inds], feat[g_inds], type='euclidean')

    if cfg.REID.VIS:
        with measure_time('Visualizing...', verbose=verbose):
            # query-gallery distance
            visualize(
                q_g_dist,
                json_dataset,
                query_ids=ids[q_inds],
                gallery_ids=ids[g_inds],
                query_cams=cams[q_inds],
                gallery_cams=cams[g_inds],
                query_paths=im_paths[q_inds],
                gallery_paths=im_paths[g_inds])

    with measure_time('Computing scores...', verbose=verbose):
        mAP, cmc_scores = compute_score(q_g_dist)

    print('{:<30}'.format('Single Query:'), end='')
    print_scores(mAP, cmc_scores)

    ###############
    # Multi Query #
    ###############

    mq_mAP, mq_cmc_scores = None, None
    if any(mq_inds):
        mq_ids = ids[mq_inds]
        mq_cams = cams[mq_inds]
        mq_feat = feat[mq_inds]
        unique_mq_ids_cams = defaultdict(list)
        for ind, (id, cam) in enumerate(zip(mq_ids, mq_cams)):
            unique_mq_ids_cams[(id, cam)].append(ind)
        keys = unique_mq_ids_cams.keys()
        assert pool_type in ['average', 'max']
        pool = np.mean if pool_type == 'average' else np.max
        mq_feat = np.stack(
            [pool(mq_feat[unique_mq_ids_cams[k]], axis=0) for k in keys])

        with measure_time(
                'Multi Query, Computing distance...', verbose=verbose):
            # multi_query-gallery distance
            mq_g_dist = compute_dist(mq_feat, feat[g_inds], type='euclidean')

        with measure_time('Multi Query, Computing scores...', verbose=verbose):
            mq_mAP, mq_cmc_scores = compute_score(
                mq_g_dist,
                query_ids=np.array(zip(*keys)[0]),
                gallery_ids=ids[g_inds],
                query_cams=np.array(zip(*keys)[1]),
                gallery_cams=cams[g_inds])

        print('{:<30}'.format('Multi Query:'), end='')
        print_scores(mq_mAP, mq_cmc_scores)

    if to_re_rank:

        ##########################
        # Re-ranked Single Query #
        ##########################

        with measure_time('Re-ranking distance...', verbose=verbose):
            # query-query distance
            q_q_dist = compute_dist(
                feat[q_inds], feat[q_inds], type='euclidean')
            # gallery-gallery distance
            g_g_dist = compute_dist(
                feat[g_inds], feat[g_inds], type='euclidean')
            # re-ranked query-gallery distance
            re_r_q_g_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        with measure_time(
                'Computing scores for re-ranked distance...', verbose=verbose):
            mAP, cmc_scores = compute_score(re_r_q_g_dist)

        print('{:<30}'.format('Re-ranked Single Query:'), end='')
        print_scores(mAP, cmc_scores)

        #########################
        # Re-ranked Multi Query #
        #########################

        if any(mq_inds):
            with measure_time(
                    'Multi Query, Re-ranking distance...', verbose=verbose):
                # multi_query-multi_query distance
                mq_mq_dist = compute_dist(mq_feat, mq_feat, type='euclidean')
                # re-ranked multi_query-gallery distance
                re_r_mq_g_dist = re_ranking(mq_g_dist, mq_mq_dist, g_g_dist)

            with measure_time(
                    'Multi Query, Computing scores for re-ranked distance...',
                    verbose=verbose):
                mq_mAP, mq_cmc_scores = compute_score(
                    re_r_mq_g_dist,
                    query_ids=np.array(zip(*keys)[0]),
                    gallery_ids=ids[g_inds],
                    query_cams=np.array(zip(*keys)[1]),
                    gallery_cams=cams[g_inds])

            print('{:<30}'.format('Re-ranked Multi Query:'), end='')
            print_scores(mq_mAP, mq_cmc_scores)

    return mAP, cmc_scores, mq_mAP, mq_cmc_scores


def get_info(entry):
    im_name = os.path.basename(entry['image'])
    pid = parse_im_name(im_name, 'id')
    cam = parse_im_name(im_name, 'cam')
    # denoting whether the im is from query, gallery, or multi query set
    mark = entry['mark']
    # print(pid, cam, im_name, mark)

    im_path = entry['image']
    return pid, cam, im_name, mark, im_path


def parse_im_name(im_name, parse_type='id'):
    """Get the person id or cam from an image name."""
    assert parse_type in ('id', 'cam')
    if parse_type == 'id':
        parsed = int(im_name[:8])
    else:
        parsed = int(im_name[9:13])
    return parsed


@contextmanager
def measure_time(enter_msg, verbose=True):
    if verbose:
        st = time.time()
        print(enter_msg)
    yield
    if verbose:
        print('Done, {:.2f}s'.format(time.time() - st))


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
    """
    print('Array size: ', array1.shape, array2.shape)

    # dist = cdist(array1, array2)
    # return dist

    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = -2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat,
        query_ids=None,
        gallery_ids=None,
        query_cams=None,
        gallery_cams=None,
        topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False,
        average=True):
    """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the 
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query, topk]
      is_valid_query: numpy array with shape [num_query], containing 0's and 
        1's, whether each query is valid or not
    If `average` is `True`:
      numpy array with shape [topk]
  """
    # Ensure numpy array
    assert isinstance(distmat, np.ndarray)
    assert isinstance(query_ids, np.ndarray)
    assert isinstance(gallery_ids, np.ndarray)
    assert isinstance(query_cams, np.ndarray)
    assert isinstance(gallery_cams, np.ndarray)

    m, n = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros([m, topk])
    is_valid_query = np.zeros(m)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        is_valid_query[i] = 1
        if single_gallery_shot:
            repeat = 100
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[i, k - j] += 1
                    break
                ret[i, k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    ret = ret.cumsum(axis=1)
    if average:
        return np.sum(ret, axis=0) / num_valid_queries
    return ret, is_valid_query


def mean_ap(distmat,
            query_ids=None,
            gallery_ids=None,
            query_cams=None,
            gallery_cams=None,
            average=True):
    """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the 
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query]
      is_valid_query: numpy array with shape [num_query], containing 0's and 
        1's, whether each query is valid or not
    If `average` is `True`:
      a scalar
  """

    # -------------------------------------------------------------------------
    # The behavior of method `sklearn.average_precision` has changed since version
    # 0.19.
    # Version 0.18.1 has same results as Matlab evaluation code by Zhun Zhong
    # (https://github.com/zhunzhong07/person-re-ranking/
    # blob/master/evaluation/utils/evaluation.m) and by Liang Zheng
    # (http://www.liangzheng.org/Project/project_reid.html).
    # My current awkward solution is sticking to this older version.
    import sklearn
    cur_version = sklearn.__version__
    required_version = '0.18.1'
    if cur_version != required_version:
        print('User Warning: Version {} is required for package scikit-learn, '
              'your current version is {}. '
              'As a result, the mAP score may not be totally correct. '
              'You can try `pip uninstall scikit-learn` '
              'and then `pip install scikit-learn=={}`'.format(
                  required_version, cur_version, required_version))
    # -------------------------------------------------------------------------

    # Ensure numpy array
    assert isinstance(distmat, np.ndarray)
    assert isinstance(query_ids, np.ndarray)
    assert isinstance(gallery_ids, np.ndarray)
    assert isinstance(query_cams, np.ndarray)
    assert isinstance(gallery_cams, np.ndarray)

    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = np.zeros(m)
    is_valid_query = np.zeros(m)
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        is_valid_query[i] = 1
        aps[i] = average_precision_score(y_true, y_score)
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    if average:
        return float(np.sum(aps)) / np.sum(is_valid_query)
    return aps, is_valid_query


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate([
        np.concatenate([q_q_dist, q_g_dist], axis=1),
        np.concatenate([q_g_dist.T, g_g_dist], axis=1)
    ],
                                   axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(
        1. * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                fi_candidate]
            if len(
                    np.intersect1d(candidate_k_reciprocal_index,
                                   k_reciprocal_index)
            ) > 2. / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (
        1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def visualize(distmat,
              json_dataset,
              query_ids=None,
              gallery_ids=None,
              query_cams=None,
              gallery_cams=None,
              query_paths=None,
              gallery_paths=None):
    print(distmat.shape)

    output_dir = os.path.join(
        get_output_dir(json_dataset.name, training=False), 'vis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue

        query_path = query_paths[i]
        query_name = os.path.basename(query_path)
        # copyfile(query_path, os.path.join(output_dir, query_name))
        im_query = cv2.imread(query_path, cv2.IMREAD_COLOR)

        bs = 4
        ms = 10
        im_list = np.zeros((im_query.shape[0] + bs * 2,
                            im_query.shape[1] * 11 + ms * 2 + ms * 2 * 10, 3),
                           np.uint8)
        im_list[:, :, :] = 255
        im_list[bs:-bs, 0:im_query.shape[1], :] = im_query
        st = im_query.shape[1] + ms * 2

        gallery_paths_this = gallery_paths[indices[i]][valid]
        for j in range(10):
            gallery_path = gallery_paths_this[j]
            gallery_name = os.path.basename(gallery_path)

            # copyfile(
            # gallery_path,
            # os.path.join(
            # output_dir, '{}_{}_{}_{}_{}'.format(
            # query_name, j, y_true[j], y_score[j], gallery_name)))

            im_gallery = cv2.imread(gallery_path, cv2.IMREAD_COLOR)
            im_gallery = cv2.resize(
                im_gallery, (im_query.shape[1], im_query.shape[0]),
                interpolation=cv2.INTER_CUBIC)
            if y_true[j]:
                im_list[:, st + ms - bs:st + ms + im_gallery.shape[1] +
                        bs, :] = [0, 255, 0]
            else:
                im_list[:, st + ms - bs:st + ms + im_gallery.shape[1] +
                        bs, :] = [0, 0, 255]
            im_list[bs:-bs, st + ms:st + ms +
                    im_gallery.shape[1], :] = im_gallery
            st += im_gallery.shape[1] + 2 * ms

        cv2.imwrite(os.path.join(output_dir, query_name), im_list)
