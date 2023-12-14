import argparse

import tensorflow as tf
import numpy as np

from utils.coco_dataset_manager import *

from utils.custom_retinanet import *


# (train_dataset, test_ds), dataset_info = tfds.load(
#     "coco/2014", split=["train", "validation"], with_info=True, data_dir=None
# )

json_path = r"C:\Users\nloftus\Documents\Datasets\coco_2017_annotations\instances_train2017.json"
save_path = r"train"
num_imgs = 50
download = True

cls_list = ['person', 'cat', 'dog']

coco_ds = CocoDSManager(json_path, save_path, max_samples=num_imgs, download=download, cls_list=cls_list, yxyw_percent=False)

train_dataset = coco_ds.train_ds
val_dataset = coco_ds.val_ds
test_ds = coco_ds.test_ds

for sample in train_dataset.take(10):


        # scaled_bb = []
        print(sample)
        # print(sample["bounding_boxes"]["classes"])
        #lbl_name = [coco_ds.coco.cats[x] for x in np.asarray(sample["objects"]["label"])]

        # for box in sample["bounding_boxes"]["boxes"]:
        #     scaled_bb.append(box)
                             
        # print(scaled_bb)

        visualize_dataset(sample["images"], sample["bounding_boxes"]["boxes"], sample["bounding_boxes"]["classes"])
        #visualize_detections(image, boxes[0], cls_name, cls_prob[0][0])