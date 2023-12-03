import argparse

import tensorflow as tf
import numpy as np

from utils.coco_dataset_manager import *

import xml.etree.ElementTree as ET

import tensorflow as tf

import keras_cv

from utils.yolo_utils import *

from utils.custom_retinanet import prepare_image

from utils.nonmaxsuppression import PreBayesianNMS

from pycocotools.coco import COCO

from utils.scalabeldataloader import TestDirectoryToScalable

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs," , len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser(description="Model Trainer")

parser.add_argument("--image_dir", "-d", help="path to test image directory", default="train")
parser.add_argument("--checkpoint_path", "-p", help="path to load checkpoint", default="best_weights")
parser.add_argument("--max_iou", "-i", help="max iou", default=.2, type=float)
parser.add_argument("--min_confidence", "-c", help="min confidence", default=.1, type=float)
parser.add_argument("--cls_path", "-l", help="path to line seperated class file", default="", type=str)


args = parser.parse_args()


#Load the class lists from text, if not specified, it gets all 80 classes
if (args.cls_path == ""):
    cls_list = None
else:
    with open(args.cls_path) as f:
        cls_list = f.readlines()
        cls_list = [cls.replace("\n", "") for cls in cls_list]
num_classes = 80 if cls_list is None else len(cls_list)



nms = PreBayesianNMS("xywh", True, confidence_threshold=args.min_confidence)

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
)

model = keras_cv.models.YOLOV8Detector(
    num_classes= num_classes,
    bounding_box_format="xywh",
    backbone=backbone,
    fpn_depth=2,
    prediction_decoder=nms
)

images = load_images(args.image_dir)

print(len(images))


latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_path)

model.load_weights(latest_checkpoint).expect_partial()


logger = TestDirectoryToScalable("detections", ".mp4")
#ds = tf.data.Dataset.from_tensor_slices(images)



idx = 0
for img in images:
    #print('hi')

    # try:
        image = tf.cast(img, dtype=tf.float32)

        input_image, ratio = prepare_image(image)
        detections = model.predict(input_image)

        boxes = np.asarray(detections["boxes"][0])

        cls_prob = np.asarray(detections["cls_prob"][0])

        cls_id = np.asarray(detections["cls_idx"][0])


        
        

        print(cls_prob)

        cls_name = [cls_list[x] for x in cls_id]

        
        correct_prob = []
        for i in range(len(cls_prob)):
            correct_prob.append(cls_prob[i][cls_id[i]])


        visualize_detections(image, boxes, cls_name, correct_prob)

        logger.add_frame_entry(idx, boxes, cls_prob, cls_name, cls_id)
        idx += 1
    # except IndexError:
    #     print("NO VALID DETECTIONS")

logger.output_scalabel_detections()