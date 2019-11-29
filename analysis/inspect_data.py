import os
import sys
import itertools
import math
import logging
import json
import re
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from collections import OrderedDict

# Root Directory of the project
ROOT_DIR = os.path.abspath('../')

# import Mask RCNN
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log
from samples.coco import coco

# Load Dataset
def LoadDataset(config):
    DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'data/COCO')

    if config.NAME == 'coco':
        DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'data/COCO')
        dataset = coco.CocoDataset()
        dataset.load_coco(DATA_ROOT_DIR, 'train')

    dataset.prepare()

    print('Image Count :: {}'.format(len(dataset.image_ids)))
    print('Class Count :: {}'.format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i+1, info['name']))

    return dataset

def DisplaySample(dataset, image_id):
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

def VisualizeObject(dataset, image_id):
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    bbox = utils.extract_bboxes(mask)

    print('image_id ::', image_id, dataset.image_reference(image_id))
    log('image', image)
    log('mask', mask)
    log('class_ids', class_ids)
    log('bbox', bbox)

    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

def ResizeImage(dataset, image_id):
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape

    image, window, scale, padding, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding)
    bbox = utils.extract_bboxes(mask)

    print('image_id ::', image_id, dataset.image_reference(image_id))
    log('image', image)
    log('mask', mask)
    log('class_ids', class_ids)
    log('bbox', bbox)

    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

def MiniMask(dataset, config, image_id):
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(dataset, config, image_id, augment=True, use_mini_mask=True)

    log('image', image)
    log('image_meta', image_meta)
    log('class_ids', class_ids)
    log('bbox', bbox)
    log('mask', mask)

    display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])

    mask = utils.expand_mask(bbox, mask, image.shape)
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

def Anchors(dataset, config, image_id):
    # Generate Anchors
    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    print('backbone_shapes::', backbone_shapes)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                              config.RPN_ANCHOR_RATIOS,
                                              backbone_shapes,
                                              config.BACKBONE_STRIDES,
                                              config.RPN_ANCHOR_STRIDE)

    # Print summary of anchors
    num_levels = len(backbone_shapes)
    anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
    print("Count: ", anchors.shape, anchors.shape[0], anchors[0])
    print("Scales: ", config.RPN_ANCHOR_SCALES)
    print("ratios: ", config.RPN_ANCHOR_RATIOS)
    print("Anchors per Cell: ", anchors_per_cell)
    print("Levels: ", num_levels)

    anchors_per_level = []
    for i in range(num_levels):
        num_cells = backbone_shapes[i][0] * backbone_shapes[i][1]
        anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)

    image, image_meta, _, _, _ = modellib.load_image_gt(dataset, config, image_id)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    count = 0
    for level in range(num_levels):
        colors = visualize.random_colors(num_levels)
        # Compute the index of the anchors at the center of the image
        level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
        level_anchors = anchors[level_start : level_start + anchors_per_level[level]]
        print("Level {}. Anchors: {:6} Feature map Shape: {}".format(level, level_anchors.shape[0], backbone_shapes[level]))

        center_cell = backbone_shapes[level] // 2
        center_cell_index = (center_cell[0] * backbone_shapes[level][1] + center_cell[1])
        level_center = center_cell_index * anchors_per_cell
        center_anchor = anchors_per_cell * ((center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE**2) + center_cell[1] / config.RPN_ANCHOR_STRIDE)
        level_center = int(center_anchor)

        # Draw anchors. Brightness show the order in the array, dark to bright.
        for i, rect in enumerate(level_anchors[level_center:level_center+anchors_per_cell]):
            y1, x1, y2, x2 = rect
            p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
                                  edgecolor=(i+1)*np.array(colors[level]) / anchors_per_cell)
            ax.add_patch(p)
            count += 1

    print('Count ::', count)
    plt.show()

def DataGenerator(dataset, image_id):
    # Generate Anchors
    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    print('backbone_shapes::', backbone_shapes)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                              config.RPN_ANCHOR_RATIOS,
                                              backbone_shapes,
                                              config.BACKBONE_STRIDES,
                                              config.RPN_ANCHOR_STRIDE)

    random_rois = 2000
    g = modellib.data_generator(dataset, config, shuffle=True, random_rois=random_rois, batch_size=4, detection_targets=True)

    if random_rois:
        [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
        log("rois", rois)
        log("mrcnn_class_ids", mrcnn_class_ids)
        log("mrcnn_bbox", mrcnn_bbox)
        log("mrcnn_mask", mrcnn_mask)
    else:
        [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)

    log("gt_class_ids", gt_class_ids)
    log("gt_boxes", gt_boxes)
    log("gt_masks", gt_masks)
    log("rpn_match", rpn_match)
    log("rpn_bbox", rpn_bbox)
    print("image_id: ", image_id, dataset.image_reference(image_id))

    # Remove the last dim in mrcnn_class_ids. It's only added
    # to satisfy Keras restriction on target shape.
    mrcnn_class_ids = mrcnn_class_ids[:,:,0]

    b = 0

    # Restore original image (reverse normalization)
    sample_image = modellib.unmold_image(normalized_images[b], config)

    # Compute anchor shifts.
    indices = np.where(rpn_match[b] == 1)[0]
    refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
    log("anchors", anchors)
    log("refined_anchors", refined_anchors)

    # Get list of positive anchors
    positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
    print("Positive anchors: {}".format(len(positive_anchor_ids)))
    negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
    print("Negative anchors: {}".format(len(negative_anchor_ids)))
    neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
    print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

    # ROI breakdown by class
    for c, n in zip(dataset.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
        if n:
            print("{:23}: {}".format(c[:20], n))

    # Show positive anchors
    visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids],
                         refined_boxes=refined_anchors)
    # Show negative anchors
    visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids])
    # Show neutral anchors. They don't contribute to training.
    visualize.draw_boxes(sample_image, boxes=anchors[np.random.choice(neutral_anchor_ids, 100)])

    plt.show()


if __name__ == '__main__':
    # Set Config
    config = coco.CocoConfig()
    config.display()
    # Load dataset
    dataset = LoadDataset(config)

    image_id = random.choice(dataset.image_ids)
    ''' Display Samples '''
    # DisplaySample(dataset=dataset, image_id=image_id)
    ''' Visualize Object '''
    # VisualizeObject(dataset=dataset, image_id=image_id)
    ''' Resize Images '''
    # ResizeImage(dataset=dataset, image_id=image_id)
    ''' Mini Mask '''
    # MiniMask(dataset=dataset, config=config, image_id=image_id)
    ''' Anchors '''
    # Anchors(dataset=dataset, config=config, image_id=image_id)
    ''' Data Generator '''
    DataGenerator(dataset=dataset, image_id=image_id)
