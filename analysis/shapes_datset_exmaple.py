import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

sys.path.append(os.path.join(ROOT_DIR, 'samples/shapes'))
import shapes


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

#####################################################################################
# Config Setting
config = shapes.ShapesConfig()
config.display()
######################################################################################

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

#######################################################################################
# Datset Setting
dataset_train = shapes.ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

dataset_val = shapes.ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()
#######################################################################################

def Visualization_Dataset():
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

def Train():
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bboc', 'mrcnn_mask'])
    # model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epoch=1, layers='heads')
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=2, layers='all')

def Inference():
    model = modellib.MaskRCNN(mode='inference', config=config, model_dir=MODEL_DIR)

    model_path = model.find_last()
    print("Loading weights from ::", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))

def Evaluation():
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))
