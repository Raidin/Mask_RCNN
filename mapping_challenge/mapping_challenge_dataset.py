import sys
import os
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from mrcnn import utils

class MappingChallengeDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, load_small=False, return_coco=True):
        self.load_small = load_small

        if self.load_small:
            annotation_path = os.path.join(dataset_dir, "annotation-small.json")
        else:
            annotation_path = os.path.join(dataset_dir, ":annotation.json")

        image_dir = os.path.join(dataset_dir, "images")

        print("Annotation Path :: ", annotation_path)
        print("Image Dir :: ", image_dir)
        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        self.coco = COCO(annotation_path)
        self.image_dir = image_dir

        # Load all classes(Only Building)
        classIds = self.coco.getCatIds()

        # Load all images
        image_ids = list(self.coco.imgs.keys())

        # register classes
        for _class_id in classIds:
            self.add_class("mapping_challenge", _class_id, self.coco.loadCats(_class_id)[0]["name"])

        # register images
        for _img_id in image_ids:
            assert(os.path.exists(os.path.join(image_dir, self.coco.imgs[_img_id]['file_name'])))
            self.add_image(
                "mapping_challenge", image_id=_img_id,
                path=os.path.join(image_dir, self.coco.imgs[_img_id]['file_name']),
                width=self.coco.imgs[_img_id]["width"],
                height=self.coco.imgs[_img_id]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(imgIds=[_img_id], catIds=classIds, iscrowd=None)))

        if return_coco:
            return self.coco

def load_mask(self, image_id):
    image_info = self.image_info[image_id]
    assert image_info["source"] == "mapping_challenge"

    instance_masks = []
    class_ids = []
    annotations = self.image_info[image_id]["annotations"]

    for ann in annotations:
        class_id = self.map_source_class_id("mapping_challenge.{}".format(ann['category_id']))

        if class_id:
            m = self.annToMask(ann, image_info['height'], image_info['width'])

            if m.max() < 1:
                continue

            instance_masks.append(m)
            class_ids.append(class_id)

    if class_ids:
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids
    else:
        # Call super class to return an empty mask
        return super(MappingChallengeDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a reference for a particular image
            Ideally you this function is supposed to return a URL
            but in this case, we will simply return the image_id
        """
        return "crowdai-mapping-challenge::{}".format(image_id)
    # The following two functions are from pycocotools with a few changes.
