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


dataset_name = 'ped_attr'
dataset_split = 'train_0905'
root_dir = '/data2/shenyunhang/Dataset/'

image_names_list = []
image_labels_list = []

meta_path = os.path.join(root_dir, dataset_name, dataset_split + '.txt')
meta_file = open(meta_path, 'r')
for line in meta_file.readlines():
    line = line.strip()
    line = line.split()
    image_name = os.path.basename(line[0])
    image_label = line[1:5]

    image_names_list.append(image_name)
    image_labels_list.append(image_label)

meta_file.close()

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
    'id': 0,
    'name': 'front',
    'supercategory': None,
}, {
    'id': 1,
    'name': 'left',
    'supercategory': None,
}, {
    'id': 2,
    'name': 'back',
    'supercategory': None,
}, {
    'id': 3,
    'name': 'right',
    'supercategory': None,
}]

coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}

image_id = 1
segmentation_id = 1
image_num = len(image_names_list)

image_break_num = 0
image_break_list = []

for image_idx, image_name in enumerate(image_names_list):
    print image_id, '/', image_num, image_name
    image_filename = os.path.join(root_dir, dataset_name, dataset_split,
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

    image_label = image_labels_list[image_idx]

    classes = [
        class_idx for class_idx, class_val in enumerate(image_label)
        if int(class_val) == 1
    ]

    if len(classes) != 1:
        image_break_num = image_break_num + 1
        image_break_list.append(image_filename)
        continue

    class_id = classes[0]
    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
    # binary_mask = np.asarray(Image.open(annotation_filename)
    # .convert('1')).astype(np.uint8)
    binary_mask = np.ones((image.size[1], image.size[0])).astype(np.uint8)

    annotation_info = pycococreatortools.create_annotation_info(
        segmentation_id,
        image_id,
        category_info,
        binary_mask,
        image.size,
        tolerance=2)

    if annotation_info is not None:
        attribution_id = annotation_info.pop('category_id', None)
        annotation_info['attribution_id'] = attribution_id
        annotation_info['classes_or_attributions'] = 1
        coco_output["annotations"].append(annotation_info)
    else:
        print 'no annotation_info for: ', image_idx, image_name
        exit(0)

    segmentation_id = segmentation_id + 1
    image_id = image_id + 1

    # if image_id > 1000:
        # break

json_path = os.path.join(root_dir, dataset_name, dataset_split + '.json')
# json_path = os.path.join(root_dir, dataset_name, 'debug.json')
with open(json_path, 'w') as output_json_file:
    json.dump(coco_output, output_json_file)

print 'image_break_list: ', image_break_list
print 'image_break_num: ', image_break_num
