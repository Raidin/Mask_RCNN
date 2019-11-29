import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = os.path.abspath('../')

sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.coco import coco

MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
COCO_DIR = os.path.join(ROOT_DIR, 'data/COCO')
CONFIG = coco.CocoConfig()
DEVICE = '/gpu:0' # /cpu:0 or /gpu:0
TEST_MODE = 'inference'

class InferenceConfig(CONFIG.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


CONFIG = InferenceConfig()

def get_ax(rows=1, cols=1, size_w=16, size_h=16):
    fig, ax = plt.subplots(rows, cols, figsize=(size_w, size_h))
    return fig, ax

def LoadData(dtype='val'):
    dataset = coco.CocoDataset()
    dataset.load_coco(COCO_DIR, dtype)
    dataset.prepare()
    print(' - images: {}\nClasses: {}\n {}'.format(len(dataset.image_ids), dataset.num_classes, dataset.class_names))

    return dataset

def LoadModel():
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=CONFIG)

    weights_path = COCO_MODEL_PATH
    print(' - Loading wieght path ::', weights_path)
    model.load_weights(weights_path, by_name=True)

    return model

def DisplayResult(image, result, dataset, *gt):
    print('lenght ground truth', len(gt))
    fig, ax = get_ax(rows=1, cols=3, size_w=15, size_h=5)
    fig.canvas.set_window_title('Prediction Results')

    visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'],
                                dataset.class_names, result['scores'], ax=ax[0],
                                title="Predictions")

    # Draw precision-recall curve
    AP, precisions, recalls, overlaps = utils.compute_ap(gt[0], gt[1], gt[2], result['rois'], result['class_ids'], result['scores'], result['masks'])
    print(' - precisions ::', precisions)
    print(' - recalls ::', recalls)
    print(' - AP ::', AP)
    visualize.plot_precision_recall(AP, precisions, recalls, ax=ax[1])

    # Grid of ground truth objects and their predictions
    visualize.plot_overlaps(gt[1], result['class_ids'], result['scores'], overlaps, dataset.class_names, is_subplot=True)

    plt.tight_layout()
    plt.show()

def ComputeBatchAP(image_ids, dataset):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, CONFIG, image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)

    return APs

def RPNPredictions():
    # Generate RPN training targets
    # target_rpn_match is 1 for positive anchors, -1 for negative anchors and 0 for neutral anchors.
    target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
    log("target_rpn_match", target_rpn_match)
    log("target_rpn_bbox", target_rpn_bbox)

    positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
    negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
    neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
    positive_anchors = model.anchors[positive_anchor_ix]
    negative_anchors = model.anchors[negative_anchor_ix]
    neutral_anchors = model.anchors[neutral_anchor_ix]
    log("positive_anchors", positive_anchors)
    log("negative_anchors", negative_anchors)
    log("neutral anchors", neutral_anchors)

    # Apply refinement deltas to positive anchors
    refined_anchors = utils.apply_box_deltas( positive_anchors, target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)

    # Display positive anchors before refinement (dotted) and after refinement (solid).
    visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors)

    ###########################################################################################################
    pillar = model.keras_model.get_layer('ROI').output

    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
    if nms_node is None:
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
    if nms_node is None: #TF 1.9-1.10
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

    rpn = model.run_graph([image], [
        ("rpn_class", model.keras_model.get_layer("rpn_class").output),
        ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
        ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
        ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
        ("post_nms_anchor_ix", nms_node),
        ("proposals", model.keras_model.get_layer("ROI").output),
    ])

    # Show top anchors by score (before refinement)
    limit = 100
    sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
    visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]])

    # Show top anchors with refinement. Then with clipping to image boundaries
    _, ax = get_ax(1, 2)
    pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
    refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
    refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
    visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit], refined_boxes=refined_anchors[:limit], ax=ax[0])
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])

    # Show refined anchors after non-max suppression
    ixs = rpn["post_nms_anchor_ix"][:limit]
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs])

    # Show final proposals
    # These are the same as the previous step (refined anchors after NMS) but with coordinates normalized to [0, 1] range Convert back to image coordinates for display
    h, w = CONFIG.IMAGE_SHAPE[:2]
    proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
    visualize.draw_boxes(image, refined_boxes=proposals)

    # Measure the RPN recall (percent of objects covered by anchors)
    # Here we measure recall for 3 different methods:
    # - All anchors
    # - All refined anchors
    # - Refined anchors after NMS
    iou_threshold = 0.7

    recall, positive_anchor_ids = utils.compute_recall(model.anchors, gt_bbox, iou_threshold)
    print("All Anchors ({:5})       Recall: {:.3f}  Positive anchors: {}".format(
        model.anchors.shape[0], recall, len(positive_anchor_ids)))

    recall, positive_anchor_ids = utils.compute_recall(rpn['refined_anchors'][0], gt_bbox, iou_threshold)
    print("Refined Anchors ({:5})   Recall: {:.3f}  Positive anchors: {}".format(
        rpn['refined_anchors'].shape[1], recall, len(positive_anchor_ids)))

    recall, positive_anchor_ids = utils.compute_recall(proposals, gt_bbox, iou_threshold)
    print("Post NMS Anchors ({:5})  Recall: {:.3f}  Positive anchors: {}".format(
        proposals.shape[0], recall, len(positive_anchor_ids)))


if __name__ == '__main__':
    print(" _____________Inspect Model_____________ ")
    dataset = LoadData()
    model = LoadModel()

    # image_id = random.choice(dataset.image_ids)
    image_id = 3076

    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, CONFIG, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, dataset.image_reference(image_id)))
    results = model.detect([image], verbose=1)

    is_display = True
    if is_display:
        DisplayResult(image, results[0], dataset, gt_bbox, gt_class_id, gt_mask)

    sys.exit()

    # Get input and output to classifier and mask heads.
    mrcnn = model.run_graph([image], [
        ("proposals", model.keras_model.get_layer("ROI").output),
        ("probs", model.keras_model.get_layer("mrcnn_class").output),
        ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])

    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]
    detections = mrcnn['detections'][0, :det_count]

    print("{} detections: {}".format(det_count, np.array(dataset.class_names)[det_class_ids]))

    captions = ["{} {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
                for c, s in zip(detections[:, 4], detections[:, 5])]
    visualize.draw_boxes(
        image,
        refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
        visibilities=[2] * len(detections),
        captions=captions, title="Detections")

    plt.show()
