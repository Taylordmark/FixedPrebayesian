import argparse
from typing import Any

import tensorflow as tf
import numpy as np

from utils.coco_dataset_manager import *

import tensorflow as tf
from tensorflow import keras

import keras_cv

from utils.yolo_utils import *

from utils.nonmaxsuppression import *

class ProbYolov8Detector:

    def __init__(self, num_classes=80, fpn_depth=3, backbone_name="yolo_v8_s_backbone_coco", box_format="xywh", min_confidence=.1, max_iou=.5) -> None:


        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            backbone_name # We will use yolov8 small backbone with coco weights
        )
        nms = DistributionNMS("xywh", True, confidence_threshold=min_confidence, iou_threshold=max_iou)

        self.model = keras_cv.models.YOLOV8Detector(
            num_classes= num_classes,
            bounding_box_format=box_format,
            backbone=backbone,
            fpn_depth=fpn_depth,
            prediction_decoder=nms
        )

        for layer in self.model.layers:
            if "conv" in layer.name:
                layer = tfp.layers.Convolution2DFlipout(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    data_format=layer.data_format,
                    dilation_rate=layer.dilation_rate,
                    activation=layer.activation,
                )

    def load_weights(self, checkpoint_path):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        self.model.load_weights(latest_checkpoint).expect_partial()
    

    def prepare_image(self, image):
        image, _, ratio = resize_and_pad_image(image, jitter=None)
        image = tf.keras.applications.resnet.preprocess_input(image)
        return tf.expand_dims(image, axis=0), ratio
    
    def to_detection_dict(self, detection):

        boxes = detection["boxes"][0]
        prob = detection["cls_prob"][0]
        idx = detection["cls_idx"][0]

        dets = []

        for i in range(len(boxes)):

            det_dict = {"boxes":boxes[i],
                        "prob":prob[i],
                        "cls_idx":idx[i]}
            dets.append(det_dict)

        return dets

    def __call__(self, image) -> Any:
        img = tf.cast(image, dtype=tf.float32)
        input_image, ratio = self.prepare_image(img)
        detection = self.model.predict(input_image)

        print(detection)

        return detection
