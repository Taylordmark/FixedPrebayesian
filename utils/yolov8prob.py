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

from utils.visualization_functions import visualize_multimodal_detections_and_gt

import shapely as shp

class ProbYolov8Detector:

    def __init__(self, num_classes=80, fpn_depth=3, backbone_name="yolo_v8_l_backbone_coco", box_format="xywh", 
                 min_confidence=.1, max_iou=.5, nms_fn=DistributionNMS, use_flipout=False, min_prob_diff=0.05) -> None:


        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            backbone_name # We will use yolov8 small backbone with coco weights
        )
        nms = nms_fn("xywh", True, confidence_threshold=min_confidence, iou_threshold=max_iou)

        self.model = keras_cv.models.YOLOV8Detector(
            num_classes= num_classes,
            bounding_box_format=box_format,
            backbone=backbone,
            fpn_depth=fpn_depth,
            prediction_decoder=nms
        )


        self.min_prob_diff = min_prob_diff

        if use_flipout:
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

        print("LOADING LATEST")
        print(latest_checkpoint)
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

    def generate_global_data(self, test_images, truth_labels, output_name="", minimum_iou=.8, visualize=True):

        #assert len(test_images) == len(truth_labels), "ERROR: Images and labels must be same length"
        true_boxes = []
        true_classes = []

        global_data = np.zeros((len(test_images, self.num_classes)))

        print(global_data)
        for true_val in truth_labels:
            true_boxes.append(np.asarray(true_val["boxes"]))
            true_classes.append(np.asarray(true_val["classes"]))


        for i in range(len(test_images)):

            img = test_images[i]
            detections = self.__call__(img)

            boxes = np.asarray(detections["boxes"])
            cls_prob = np.asarray(detections["cls_prob"])
            cls_id =  np.asarray(detections["cls_ids"])

            valid_idx = []


            for j in range(len(boxes)):
                box = boxes[j]

                dt_box:shp.box = shp.box(box[0], box[1], box[0]+box[2], box[1]+box[3])
    

                for k in range(len(true_boxes[i])):



                    tox = true_boxes[i][k]

                    tr_box = shp.box(tox[0], tox[1], tox[0]+tox[2], tox[1]+tox[3])


                    intersect:shp.Polygon = shp.intersection(dt_box, tr_box)
                    union = shp.union(dt_box, tr_box)

                    if intersect.is_empty:
                        continue

                    iou = intersect.area / union.area

                    #print(f"{iou}  {cls_id[j]} {true_classes[i][k]}")

                
                    if iou > minimum_iou and true_classes[i][k] in cls_id[j]:
                        #print(iou)
                        valid_idx.append((j,k))

                        # cls_idx = cls_id[j]

                        # global_data[cl][i] = cls_prob[i]

                    

            show_trs = []
            show_trcls = []
            show_prob = []


            show_gts =[]
            show_gtcls = []

            for pair in valid_idx:
                j, k = pair
                show_trs.append(boxes[j])
                show_trcls.append(cls_id[j])
                show_prob.append(cls_prob[j])

                show_gts.append(true_boxes[i][k])
                show_gtcls.append(true_classes[i][k])

            if (show_trs != [] and show_gts != [] and visualize):
                visualize_multimodal_detections_and_gt(img, show_trs, show_trcls, show_prob, show_gts, show_gtcls)
               

            

            


    def __call__(self, image) -> Any:
        img = tf.cast(image, dtype=tf.float32)
        input_image, ratio = self.prepare_image(img)
        detection = self.model.predict(input_image)

        cls_prob = detection['cls_prob'][0]

        cls_ids = []
 
        for distribs in cls_prob:

            i = 0

            ids = []

            

            min = np.min(distribs)




            for prob in distribs:

                #print(f"{prob} > {min} + {self.min_prob_diff} = {min+self.min_prob_diff}")
                if prob > min+self.min_prob_diff:
                    ids.append(i)
                i +=1

            

            if ids == []:
                ids.append(-1)
            
            cls_ids.append(ids)



        ret = {"boxes":detection['boxes'][0],
               "cls_prob":cls_prob,
               "cls_ids":cls_ids,
               "confidence":detection['confidence'][0]}


        return ret
