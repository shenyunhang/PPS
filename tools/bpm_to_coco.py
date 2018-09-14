import datetime
import json
import os

import os.path as osp
import cPickle as pickle
import numpy as np

from PIL import Image
from pycococreatortools import pycococreatortools


def load_pickle(path):
    """Check and load pickle object.
    According to this post: https://stackoverflow.com/a/41733927, cPickle and 
    disabling garbage collector helps with loading speed."""
    print path
    assert osp.exists(path)
    # gc.disable()
    with open(path, 'rb') as f:
        ret = pickle.load(f)
        # gc.enable()
    return ret


root_dir = os.path.expanduser('~/Dataset')

for dataset_name in [
        'market1501', 'duke', 'cuhk03/labeled', 'cuhk03/detected'
]:
    for dataset_split in ['trainval', 'test']:
        print('=' * 100)
        print(dataset_name, dataset_split)
        im_names_key = dataset_split + '_im_names'
        ids2labels_key = dataset_split + '_ids2labels'
        marks_key = dataset_split + '_marks'

        pkl_path = os.path.join(root_dir, dataset_name, 'partitions.pkl')
        pkl = load_pickle(pkl_path)

        im_names_list = pkl[im_names_key]
        image_num = len(im_names_list)

        if ids2labels_key not in pkl.keys():
            # ids2labels_dict = {i: i for i in range(image_num)}
            ids2labels_dict = dict()
            for image_idx, image_name in enumerate(im_names_list):
                im_name = os.path.basename(image_name)
                pid = int(im_name[:8])
                if pid not in ids2labels_dict.keys():
                    ids2labels_dict[pid] = len(ids2labels_dict.keys())
        else:
            ids2labels_dict = pkl[ids2labels_key]

        if marks_key not in pkl.keys():
            marks_list = [1 for _ in range(len(im_names_list))]
        else:
            marks_list = pkl[marks_key]

        print pkl['trainval_ids2labels'].items()[:100]
        print len(pkl['trainval_ids2labels'].keys())
        print pkl['trainval_im_names'][:100]
        print len(pkl['val_im_names'])
        print len(pkl['trainval_im_names'])
        print len(pkl['test_im_names'])
        print len(pkl['val_marks'])
        print pkl.keys()

        INFO = {
            "description": "Example Dataset",
            "url": "https://github.com/waspinator/pycococreator",
            "version": "0.1.0",
            "year": 2018,
            "contributor": "waspinator",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        LICENSES = [{
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }]

        CATEGORIES = [{
            'id': v,
            'name': str(k),
            'supercategory': None,
        } for k, v in ids2labels_dict.items()]

        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

        image_id = 1
        segmentation_id = 1

        image_break_num = 0
        image_break_list = []

        for image_idx, image_name in enumerate(im_names_list):
            if image_idx % 1000 == 0:
                print dataset_name, dataset_split, image_id, '/', image_num, image_name
            image_filename = os.path.join(root_dir, dataset_name, 'images',
                                          image_name)
            try:
                image = Image.open(image_filename)
            except IOError:
                image_break_num = image_break_num + 1
                image_break_list.append(image_filename)
                continue
                pass

            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            person_id = int(image_name[:8])

            class_id = ids2labels_dict[person_id]
            category_info = {
                'id': class_id,
                'is_crowd': 'crowd' in image_filename
            }
            # binary_mask = np.asarray(Image.open(annotation_filename)
            # .convert('1')).astype(np.uint8)
            binary_mask = np.ones((image.size[1],
                                   image.size[0])).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id,
                image_id,
                category_info,
                binary_mask,
                image.size,
                tolerance=2)

            if annotation_info is not None:
                annotation_info['classes_or_attributions'] = 0
                annotation_info['mark'] = marks_list[image_idx]
                coco_output["annotations"].append(annotation_info)
            else:
                print 'no annotation_info for: ', image_idx, image_name
                exit(0)

            segmentation_id = segmentation_id + 1
            image_id = image_id + 1

            # if image_id > 1000:
            # break

        json_path = os.path.join(root_dir, dataset_name,
                                 dataset_split + '.json')
        # json_path = os.path.join(root_dir, dataset_name, 'debug.json')
        with open(json_path, 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

        print 'total id: ', len(ids2labels_dict)
        print 'image_break_list: ', image_break_list
        print 'image_break_num: ', image_break_num
