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
num_imgs = 10
download = True

cls_list = ['person', 'cat', 'dog', 'potted plant']

coco_ds = CocoDSManager(json_path, save_path, max_samples=num_imgs, download=download, cls_list=cls_list)

train_dataset = coco_ds.train_ds
val_dataset = coco_ds.val_ds
test_ds = coco_ds.test_ds

for sample in test_ds.take(3):


        scaled_bb = []
        print(sample["objects"]["label"])
        #lbl_name = [coco_ds.coco.cats[x] for x in np.asarray(sample["objects"]["label"])]

        for box in sample["objects"]["bbox"]:
            print(box)
            scaled_b = xywh_to_yxyx_percent(box, np.asarray(sample["image"]).shape)
            print(scaled_b)
            scaled_bb.append(yxyx_percent_to_xywh(box, np.asarray(sample["image"]).shape))
                             
        print(scaled_bb)

        visualize_dataset(sample["image"], scaled_bb, sample["objects"]["label"])
        #visualize_detections(image, boxes[0], cls_name, cls_prob[0][0])