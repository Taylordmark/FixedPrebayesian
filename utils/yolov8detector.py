import argparse
from typing import Any

import tensorflow as tf
import numpy as np

from utils.coco_dataset_manager import *

import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras

import keras_cv

from utils.yolo_utils import *

from utils.custom_retinanet import prepare_image

from utils.nonmaxsuppression import *

class Yolov8Detector:

    def __init__(self, weights_path, num_classes=80, fpn_depth=2, backbone_name="yolo_v8_s_backbone_coco", box_format="xywh", min_confidence=.1, max_iou=.5) -> None:


        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
        )
        nms = PreBayesianNMS("xywh", True, confidence_threshold=min_confidence, iou_threshold=max_iou)

        self.model = keras_cv.models.YOLOV8Detector(
            num_classes= num_classes,
            bounding_box_format=box_format,
            backbone=backbone,
            fpn_depth=fpn_depth,
            prediction_decoder=nms
        )

        latest_checkpoint = tf.train.latest_checkpoint(weights_path)
    
        self.model.load_weights(latest_checkpoint).expect_partial()

    def __call__(self, image) -> Any:
        return self.model.predict(image)
