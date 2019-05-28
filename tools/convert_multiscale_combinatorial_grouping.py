"""Script to convert Mutiscale Combinatorial Grouping proposal boxes into the Detectron proposal
file format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import numpy as np
import scipy.io as sio
import sys
import os

from detectron.datasets.json_dataset import JsonDataset

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    dir_in = sys.argv[2]
    file_out = sys.argv[3]

    ds = JsonDataset(dataset_name)
    roidb = ds.get_roidb()

    boxes = []
    scores = []
    ids = []
    for i in range(len(roidb)):
        if i % 1000 == 0:
            print('{}/{}'.format(i + 1, len(roidb)))

        index = os.path.splitext(os.path.basename(roidb[i]['image']))[0]
        box_file = os.path.join(dir_in, '{}.mat'.format(index))
        boxes_data = sio.loadmat(box_file)['bboxes']
        scores_data = sio.loadmat(box_file)['bboxes_scores']
        assert boxes_data.shape[0] == scores_data.shape[0]
        # selective search boxes are 1-indexed and (y1, x1, y2, x2)
        # Boxes from the MCG website are in (y1, x1, y2, x2) order
        # boxes_data = boxes_data[:, (1, 0, 3, 2)] - 1
        boxes_data_ = boxes_data.astype(np.uint16) - 1
        boxes_data = boxes_data_[:, (1, 0, 3, 2)]
        boxes.append(boxes_data.astype(np.uint16))
        scores.append(scores_data.astype(np.float32))
        ids.append(roidb[i]['id'])

    with open(file_out, 'wb') as f:
        pickle.dump(
            dict(boxes=boxes, scores=scores, indexes=ids), f,
            pickle.HIGHEST_PROTOCOL)
