from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import numpy.random as npr
import random
import os
import cv2
import math

from detectron.core.config import cfg
import detectron.modeling.FPN as fpn
import detectron.roi_data.keypoint_rcnn as keypoint_rcnn_roi_data
import detectron.roi_data.mask_rcnn as mask_rcnn_roi_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)


def get_reid_blob_names(is_training=True):

    blob_names = []
    if is_training:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_int32']
        blob_names += ['labels_oh']
        if cfg.REID.PSE_ON:
            blob_names += ['attr_labels_int32']
            blob_names += ['weight']
            blob_names += ['attr_weight']
    return blob_names
    """Fast R-CNN blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois']
    if is_training:
        # bbox_targets blob: R bounding-box regression targets with 4
        # targets per class
        blob_names += ['bbox_targets']
        # bbox_inside_weights blob: At most 4 targets per roi are active
        # this binary vector sepcifies the subset of active targets
        blob_names += ['bbox_inside_weights']
        blob_names += ['bbox_outside_weights']
    if is_training and cfg.MODEL.MASK_ON:
        # 'mask_rois': RoIs sampled for training the mask prediction branch.
        # Shape is (#masks, 5) in format (batch_idx, x1, y1, x2, y2).
        blob_names += ['mask_rois']
        # 'roi_has_mask': binary labels for the RoIs specified in 'rois'
        # indicating if each RoI has a mask or not. Note that in some cases
        # a *bg* RoI will have an all -1 (ignore) mask associated with it in
        # the case that no fg RoIs can be sampled. Shape is (batchsize).
        blob_names += ['roi_has_mask_int32']
        # 'masks_int32' holds binary masks for the RoIs specified in
        # 'mask_rois'. Shape is (#fg, M * M) where M is the ground truth
        # mask size.
        blob_names += ['masks_int32']
    if is_training and cfg.MODEL.KEYPOINTS_ON:
        # 'keypoint_rois': RoIs sampled for training the keypoint prediction
        # branch. Shape is (#instances, 5) in format (batch_idx, x1, y1, x2,
        # y2).
        blob_names += ['keypoint_rois']
        # 'keypoint_locations_int32': index of keypoint in
        # KRCNN.HEATMAP_SIZE**2 sized array. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_locations_int32']
        # 'keypoint_weights': weight assigned to each target in
        # 'keypoint_locations_int32'. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_weights']
        # 'keypoint_loss_normalizer': optional normalization factor to use if
        # cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
        blob_names += ['keypoint_loss_normalizer']
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_fpn' + str(lvl)]
        blob_names += ['rois_idx_restore_int32']
        if is_training:
            if cfg.MODEL.MASK_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['mask_rois_fpn' + str(lvl)]
                blob_names += ['mask_rois_idx_restore_int32']
            if cfg.MODEL.KEYPOINTS_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['keypoint_rois_fpn' + str(lvl)]
                blob_names += ['keypoint_rois_idx_restore_int32']
    return blob_names


def add_reid_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_rois(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

    return True

    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs)

    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True
    if cfg.MODEL.KEYPOINTS_ON:
        valid = keypoint_rcnn_roi_data.finalize_keypoint_minibatch(
            blobs, valid)

    return valid


def _sample_rois(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    if cfg.REID.PSE_ON:
        img_labels = np.array([0], dtype=np.float32)
        attr_img_labels = np.array([0], dtype=np.float32)
        weight = np.array([0.0], dtype=np.float32)
        attr_weight = np.array([0.0], dtype=np.float32)

        gt_inds = np.where(roidb['gt_classes'] > 0)[0]
        assert len(gt_inds) <= 2, 'Only one ground truth for image is allowed.'
        gt_classes = roidb['gt_classes'][gt_inds].copy()

        gt_inds = np.where(roidb['gt_attributions'] > 0)[0]
        assert len(gt_inds) <= 2, 'Only one ground truth for image is allowed.'
        gt_attributions = roidb['gt_attributions'][gt_inds].copy()

        classes_or_attributions = roidb['classes_or_attributions']
        for i in range(len(gt_classes)):
            if classes_or_attributions[i] == 0:
                img_labels[0] = gt_classes[i] - 1
                weight[0] = 1.0
            elif classes_or_attributions[i] == 1:
                attr_img_labels[0] = gt_attributions[i] - 1
                attr_weight[0] = cfg.REID.PSE_WEIGHT
            else:
                img_labels[0] = gt_classes[i] - 1
                weight[0] = 1.0
                attr_img_labels[0] = gt_attributions[i] - 1
                attr_weight[0] = cfg.REID.PSE_WEIGHT
        blob_dict = dict(
            labels_int32=img_labels.astype(np.int32, copy=False),
            attr_labels_int32=attr_img_labels.astype(np.int32, copy=False),
            weight=weight.astype(np.float32, copy=False),
            attr_weight=attr_weight.astype(np.float32, copy=False),
        )
        return blob_dict

    # Get image label
    img_labels_oh = np.zeros((1, cfg.MODEL.NUM_CLASSES - 1), dtype=np.float32)
    img_labels = np.zeros((1), dtype=np.float32)

    gt_inds = np.where(roidb['gt_classes'] > 0)[0]
    assert len(gt_inds) == 1, 'Only one ground truth for image is allowed.'
    gt_classes = roidb['gt_classes'][gt_inds].copy()

    img_labels_oh[0][gt_classes[0] - 1] = 1
    img_labels[0] = gt_classes[0] - 1

    blob_dict = dict(
        labels_int32=img_labels.astype(np.int32, copy=False),
        labels_oh=img_labels_oh.astype(np.float32, copy=False),
    )
    return blob_dict

    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
    max_overlaps = roidb['max_overlaps']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
            fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
            bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Label is the class each RoI has max overlap with
    sampled_labels = roidb['max_classes'][keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes = roidb['boxes'][keep_inds]

    bbox_targets, bbox_inside_weights = _expand_bbox_targets(
        roidb['bbox_targets'][keep_inds, :])
    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones(
        (sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights)

    # Optionally add Mask R-CNN blobs
    if cfg.MODEL.MASK_ON:
        mask_rcnn_roi_data.add_mask_rcnn_blobs(blob_dict, sampled_boxes, roidb,
                                               im_scale, batch_idx)

    # Optionally add Keypoint R-CNN blobs
    if cfg.MODEL.KEYPOINTS_ON:
        keypoint_rcnn_roi_data.add_keypoint_rcnn_blobs(
            blob_dict, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx)

    return blob_dict


def random_crop(im):
    # Randomly crop a sub-image.
    crop_prob = cfg.REID.CROP_PROB
    crop_ratio = cfg.REID.CROP_RATIO
    assert crop_prob <= 1
    assert crop_prob >= 0
    if crop_prob == 0 or np.random.uniform() > crop_prob:
        return im, [0, 0, im.shape[0] - 1, im.shape[1] - 1]
    assert crop_ratio > 0
    assert crop_ratio < 1
    h_ratio = np.random.uniform(crop_ratio, 1)
    w_ratio = np.random.uniform(crop_ratio, 1)
    crop_h = int(im.shape[0] * h_ratio)
    crop_w = int(im.shape[1] * w_ratio)
    h_start = np.random.randint(0, im.shape[0] - crop_h)
    w_start = np.random.randint(0, im.shape[1] - crop_w)
    im = np.copy(im[h_start:h_start + crop_h, w_start:w_start + crop_w, :])

    im_crop = [h_start, w_start, h_start + crop_h - 1, w_start + crop_w - 1]
    return im, im_crop


def horizontal_crop(im):
    horizontal_crop_prob = cfg.REID.HORIZONTAL_CROP_PROB
    horizontal_crop_ratio = cfg.REID.HORIZONTAL_CROP_RATIO
    # Horizontal Crop
    if ((horizontal_crop_ratio < 1) and (horizontal_crop_prob > 0)
            and (np.random.uniform() < horizontal_crop_prob)
            and im.shape[0] * 1.0 / im.shape[1] > 1.5):
        h_ratio = np.random.uniform(horizontal_crop_ratio, 1)
        crop_h = int(im.shape[0] * h_ratio)
        im = im[0:crop_h]

        return im, [0, 0, crop_h - 1, im.shape[1] - 1]
    else:
        return im, [0, 0, im.shape[0] - 1, im.shape[1] - 1]


# Do not use it
# padding image to 3:1 before resizing
def fix_rate(img):
    if not cfg.REID.FIX_RATE:
        return img

    h = img.shape[0]
    w = img.shape[1]
    #print "h:%d,w:%d" % (h, w)
    if float(h) / float(w) <= 3:
        out_img = np.random.uniform(0, 1, size=(3 * w, w, 3))
        out_img[0:3 * w, 0:w, 0] = self.im_mean[0] * 255
        out_img[0:3 * w, 0:w, 1] = self.im_mean[1] * 255
        out_img[0:3 * w, 0:w, 2] = self.im_mean[2] * 255
    else:
        out_img = np.random.uniform(0, 1, size=(h, int(h / 3), 3))
        out_img[0:h, 0:int(h / 3), 0] = self.im_mean[0] * 255
        out_img[0:h, 0:int(h / 3), 1] = self.im_mean[1] * 255
        out_img[0:h, 0:int(h / 3), 2] = self.im_mean[2] * 255
    h_o = out_img.shape[0]
    w_o = out_img.shape[1]
    delta_h = (h_o - h) / 2
    delta_w = (w_o - w) / 2
    #print "delta_h: %d,delta_w: %d" % (delta_h,delta_w)
    #print "h_o:%d w_o: %d" % (h_o,w_o)
    out_img[delta_h:h + delta_h, delta_w:w + delta_w, :] = img
    return out_img


def hsv_jitter(im):
    hsv_jitter_prob = cfg.REID.HSV_JITTER_PROB
    assert hsv_jitter_prob <= 1
    assert hsv_jitter_prob >= 0

    if hsv_jitter_prob == 0 or np.random.uniform() > hsv_jitter_prob:
        return im

    saturation_range = cfg.REID.SATURATION_RANGE
    hue_range = cfg.REID.HUE_RANGE
    value_range = cfg.REID.VALUE_RANGE

    im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int)
    #saturation
    if saturation_range > 0:
        offset = np.random.randint(-saturation_range, saturation_range)
        #print offset
        im_hsv[:, :, 1] = im_hsv[:, :, 1] + offset

    #hue
    if hue_range > 0:
        offset = np.random.randint(-hue_range, hue_range)
        #print offset
        im_hsv[:, :, 0] = im_hsv[:, :, 0] + offset

    #value
    if value_range > 0:
        offset = np.random.randint(-value_range, value_range)
        #print offset
        im_hsv[:, :, 2] = im_hsv[:, :, 2] + offset

    im_hsv = np.clip(im_hsv, 0, 255).astype(np.uint8)
    im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
    return im_rgb


def gaussian_blur(im):
    gaussian_blur_prob = cfg.REID.GAUSSIAN_BLUR_PROB
    gaussian_blur_kernel = cfg.REID.GAUSSIAN_BLUR_KERNEL
    if gaussian_blur_prob == 0 or np.random.uniform() > gaussian_blur_prob:
        return im

    sizes = range(1, gaussian_blur_kernel, 2)
    kernel_size = random.sample(sizes, 1)[0]
    im = cv2.GaussianBlur(im, (kernel_size, kernel_size), 0)
    return im


def random_erasing(img):
    random_erasing_prob = cfg.REID.RANDOM_ERASING_PROB
    sl = cfg.REID.SL
    sh = cfg.REID.SH
    r1 = cfg.REID.R1
    if random_erasing_prob == 0 or np.random.uniform(0,
                                                     1) > random_erasing_prob:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1.0 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        #rate = 1
        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[x1:x1 + h, y1:y1 + w, 0] = cfg.PIXEL_MEANS[0, 0, 0]
                img[x1:x1 + h, y1:y1 + w, 1] = cfg.PIXEL_MEANS[0, 0, 1]
                img[x1:x1 + h, y1:y1 + w, 2] = cfg.PIXEL_MEANS[0, 0, 2]
            else:
                img[x1:x1 + h, y1:y1 + w, 0] = cfg.PIXEL_MEANS[0, 0, 0]
            return img
    return img


def save_image(img, name):
    output_dir = cfg.OUTPUT_DIR

    cv2.imwrite(os.path.join(output_dir, name + '.png'), img)


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def _add_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        target_lvls = fpn.map_rois_to_fpn_levels(blobs[rois_blob_name][:, 1:5],
                                                 lvl_min, lvl_max)
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn.add_multilevel_roi_blobs(blobs, rois_blob_name,
                                     blobs[rois_blob_name], target_lvls,
                                     lvl_min, lvl_max)

    _distribute_rois_over_fpn_levels('rois')
    if cfg.MODEL.MASK_ON:
        _distribute_rois_over_fpn_levels('mask_rois')
    if cfg.MODEL.KEYPOINTS_ON:
        _distribute_rois_over_fpn_levels('keypoint_rois')
