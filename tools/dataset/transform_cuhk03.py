"""Refactor file directories, save/rename images and partition the 
train/val/test set, in order to support the unified dataset interface.
"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')

from zipfile import ZipFile
import os.path as osp
import sys
import h5py
from scipy.misc import imsave
from itertools import chain


import os
import cPickle as pickle


def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """
  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)


def load_pickle(path):
  """Check and load pickle object.
  According to this post: https://stackoverflow.com/a/41733927, cPickle and 
  disabling garbage collector helps with loading speed."""
  assert osp.exists(path)
  # gc.disable()
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  # gc.enable()
  return ret


def save_pickle(obj, path):
  """Create dir and save file."""
  may_make_dir(osp.dirname(osp.abspath(path)))
  with open(path, 'wb') as f:
    pickle.dump(obj, f, protocol=2)


new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'

def parse_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(im_name[:8])
  else:
    parsed = int(im_name[9:13])
  return parsed


import numpy as np
import glob
from collections import defaultdict
import shutil


def get_im_names(im_dir, pattern='*.jpg', return_np=True, return_path=False):
  """Get the image names in a dir. Optional to return numpy array, paths."""
  im_paths = glob.glob(osp.join(im_dir, pattern))
  im_names = [osp.basename(path) for path in im_paths]
  ret = im_paths if return_path else im_names
  if return_np:
    ret = np.array(ret)
  return ret


def move_ims(ori_im_paths, new_im_dir, parse_im_name, new_im_name_tmpl):
  """Rename and move images to new directory."""
  cnt = defaultdict(int)
  new_im_names = []
  for im_path in ori_im_paths:
    im_name = osp.basename(im_path)
    id = parse_im_name(im_name, 'id')
    cam = parse_im_name(im_name, 'cam')
    cnt[(id, cam)] += 1
    new_im_name = new_im_name_tmpl.format(id, cam, cnt[(id, cam)] - 1)
    shutil.copy(im_path, osp.join(new_im_dir, new_im_name))
    new_im_names.append(new_im_name)
  return new_im_names


def partition_train_val_set(im_names, parse_im_name,
                            num_val_ids=None, val_prop=None, seed=1):
  """Partition the trainval set into train and val set. 
  Args:
    im_names: trainval image names
    parse_im_name: a function to parse id and camera from image name
    num_val_ids: number of ids for val set. If not set, val_prob is used.
    val_prop: the proportion of validation ids
    seed: the random seed to reproduce the partition results. If not to use, 
      then set to `None`.
  Returns:
    a dict with keys (`train_im_names`, 
                      `val_query_im_names`, 
                      `val_gallery_im_names`)
  """
  np.random.seed(seed)
  # Transform to numpy array for slicing.
  if not isinstance(im_names, np.ndarray):
    im_names = np.array(im_names)
  np.random.shuffle(im_names)
  ids = np.array([parse_im_name(n, 'id') for n in im_names])
  cams = np.array([parse_im_name(n, 'cam') for n in im_names])
  unique_ids = np.unique(ids)
  np.random.shuffle(unique_ids)

  # Query indices and gallery indices
  query_inds = []
  gallery_inds = []

  if num_val_ids is None:
    assert 0 < val_prop < 1
    num_val_ids = int(len(unique_ids) * val_prop)
  num_selected_ids = 0
  for unique_id in unique_ids:
    query_inds_ = []
    # The indices of this id in trainval set.
    inds = np.argwhere(unique_id == ids).flatten()
    # The cams that this id has.
    unique_cams = np.unique(cams[inds])
    # For each cam, select one image for query set.
    for unique_cam in unique_cams:
      query_inds_.append(
        inds[np.argwhere(cams[inds] == unique_cam).flatten()[0]])
    gallery_inds_ = list(set(inds) - set(query_inds_))
    # For each query image, if there is no same-id different-cam images in
    # gallery, put it in gallery.
    for query_ind in query_inds_:
      if len(gallery_inds_) == 0 \
          or len(np.argwhere(cams[gallery_inds_] != cams[query_ind])
                     .flatten()) == 0:
        query_inds_.remove(query_ind)
        gallery_inds_.append(query_ind)
    # If no query image is left, leave this id in train set.
    if len(query_inds_) == 0:
      continue
    query_inds.append(query_inds_)
    gallery_inds.append(gallery_inds_)
    num_selected_ids += 1
    if num_selected_ids >= num_val_ids:
      break

  query_inds = np.hstack(query_inds)
  gallery_inds = np.hstack(gallery_inds)
  val_inds = np.hstack([query_inds, gallery_inds])
  trainval_inds = np.arange(len(im_names))
  train_inds = np.setdiff1d(trainval_inds, val_inds)

  train_inds = np.sort(train_inds)
  query_inds = np.sort(query_inds)
  gallery_inds = np.sort(gallery_inds)

  partitions = dict(train_im_names=im_names[train_inds],
                    val_query_im_names=im_names[query_inds],
                    val_gallery_im_names=im_names[gallery_inds])

  return partitions


def save_images(mat_file, save_dir, new_im_name_tmpl):
  def deref(mat, ref):
    return mat[ref][:].T

  def dump(mat, refs, pid, cam, im_dir):
    """Save the images of a person under one camera."""
    for i, ref in enumerate(refs):
      im = deref(mat, ref)
      if im.size == 0 or im.ndim < 2: break
      fname = new_im_name_tmpl.format(pid, cam, i)
      imsave(osp.join(im_dir, fname), im)

  mat = h5py.File(mat_file, 'r')
  labeled_im_dir = osp.join(save_dir, 'labeled/images')
  detected_im_dir = osp.join(save_dir, 'detected/images')
  all_im_dir = osp.join(save_dir, 'all/images')

  may_make_dir(labeled_im_dir)
  may_make_dir(detected_im_dir)
  may_make_dir(all_im_dir)

  # loop through camera pairs
  pid = 0
  for labeled, detected in zip(mat['labeled'][0], mat['detected'][0]):
    labeled, detected = deref(mat, labeled), deref(mat, detected)
    assert labeled.shape == detected.shape
    # loop through ids in a camera pair
    for i in range(labeled.shape[0]):
      # We don't care about whether different persons are under same cameras,
      # we only care about the same person being under different cameras or not.
      dump(mat, labeled[i, :5], pid, 0, labeled_im_dir)
      dump(mat, labeled[i, 5:], pid, 1, labeled_im_dir)
      dump(mat, detected[i, :5], pid, 0, detected_im_dir)
      dump(mat, detected[i, 5:], pid, 1, detected_im_dir)
      dump(mat, chain(detected[i, :5], labeled[i, :5]), pid, 0, all_im_dir)
      dump(mat, chain(detected[i, 5:], labeled[i, 5:]), pid, 1, all_im_dir)
      pid += 1
      if pid % 100 == 0:
        sys.stdout.write('\033[F\033[K')
        print('Saving images {}/{}'.format(pid, 1467))


def transform(zip_file, train_test_partition_file, save_dir=None):
  """Save images and partition the train/val/test set.
  """
  print("Extracting zip file")
  root = osp.dirname(osp.abspath(zip_file))
  if save_dir is None:
    save_dir = root
  may_make_dir(save_dir)
  with ZipFile(zip_file) as z:
    z.extractall(path=save_dir)
  print("Extracting zip file done")
  mat_file = osp.join(save_dir, osp.basename(zip_file)[:-4], 'cuhk-03.mat')

  save_images(mat_file, save_dir, new_im_name_tmpl)

  if osp.exists(train_test_partition_file):
    train_test_partition = load_pickle(train_test_partition_file)
  else:
    raise RuntimeError('Train/test partition file should be provided.')

  for im_type in ['detected', 'labeled']:
    trainval_im_names = train_test_partition[im_type]['train_im_names']
    trainval_ids = list(set([parse_im_name(n, 'id')
                             for n in trainval_im_names]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    trainval_ids.sort()
    trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
    train_val_partition = \
      partition_train_val_set(trainval_im_names, parse_im_name, num_val_ids=100)
    train_im_names = train_val_partition['train_im_names']
    train_ids = list(set([parse_im_name(n, 'id')
                          for n in train_val_partition['train_im_names']]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    train_ids.sort()
    train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

    # A mark is used to denote whether the image is from
    #   query (mark == 0), or
    #   gallery (mark == 1), or
    #   multi query (mark == 2) set

    val_marks = [0, ] * len(train_val_partition['val_query_im_names']) \
                + [1, ] * len(train_val_partition['val_gallery_im_names'])
    val_im_names = list(train_val_partition['val_query_im_names']) \
                   + list(train_val_partition['val_gallery_im_names'])
    test_im_names = list(train_test_partition[im_type]['query_im_names']) \
                    + list(train_test_partition[im_type]['gallery_im_names'])
    test_marks = [0, ] * len(train_test_partition[im_type]['query_im_names']) \
                 + [1, ] * len(
      train_test_partition[im_type]['gallery_im_names'])
    partitions = {'trainval_im_names': trainval_im_names,
                  'trainval_ids2labels': trainval_ids2labels,
                  'train_im_names': train_im_names,
                  'train_ids2labels': train_ids2labels,
                  'val_im_names': val_im_names,
                  'val_marks': val_marks,
                  'test_im_names': test_im_names,
                  'test_marks': test_marks}
    partition_file = osp.join(save_dir, im_type, 'partitions.pkl')
    save_pickle(partitions, partition_file)
    print('Partition file for "{}" saved to {}'.format(im_type, partition_file))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform CUHK03 Dataset")
  parser.add_argument(
    '--zip_file',
    type=str,
    default='~/Dataset/cuhk03/cuhk03_release.zip')
  parser.add_argument(
    '--save_dir',
    type=str,
    default='~/Dataset/cuhk03')
  parser.add_argument(
    '--train_test_partition_file',
    type=str,
    default='~/Dataset/cuhk03/re_ranking_train_test_split.pkl')
  args = parser.parse_args()
  zip_file = osp.abspath(osp.expanduser(args.zip_file))
  train_test_partition_file = osp.abspath(osp.expanduser(
    args.train_test_partition_file))
  save_dir = osp.abspath(osp.expanduser(args.save_dir))
  transform(zip_file, train_test_partition_file, save_dir)
