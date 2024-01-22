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
import pickle

class ProbYolov8Detector:

    def __init__(self, num_classes=80, fpn_depth=3, backbone_name="yolo_v8_l_backbone_coco", box_format="xywh", 
                 min_confidence=.1, max_iou=.5, nms_fn=DistributionNMS, use_flipout=False, min_prob_diff=0.05) -> None:


        self.num_classes = num_classes
        self.max_iou = max_iou
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

    
    def generate_global_data(self, test_images, truth_labels, output_name="global_data", minimum_iou=.8, visualize=True, output_file=True, min_confidence=.15):
        """
        Generates a global data array based on the model, test images, and labels

        test_images: list of images, np format
        truth_labels: list of dictionaries for each image, dictionary seperates into ["boxes"] and ["classes"]
        output_name: name pickle is saved as, extension not needed
        minimum_iou: currently unused, besides for visualization
        visualize: whether to show visuals or not
        output_file: whether to save result to pickle or not

        returns: list of dictionaries, with each dictionary being global data for a different frame

        """

        true_boxes = []
        true_classes = []




        #initializes new global data dictionary
        global_data = {}
        for c in range(-1, self.num_classes):
            global_data[c] = []



        #loads true data, did it this way due to weird errors I was getting
        for true_val in truth_labels:
            true_boxes.append(np.asarray(true_val["boxes"]))
            true_classes.append(np.asarray(true_val["classes"]))


        
        for i in range(len(test_images)):
            #repeats for each test image provided

            valid_idx = []


            img = test_images[i]
            detections = self.__call__(img)

            boxes = np.asarray(detections["boxes"])



            cls_prob = np.asarray(detections["cls_prob"])
            cls_id =  detections["cls_ids"]



            #for each detection from the model
            for j in range(len(boxes)):
                box = boxes[j]

                dt_box:shp.box = shp.box(box[0], box[1], box[0]+box[2], box[1]+box[3])
    

                #for each real box
                for k in range(len(true_boxes[i])):



                    tox = true_boxes[i][k]

                    tr_box = shp.box(tox[0], tox[1], tox[0]+tox[2], tox[1]+tox[3])


                    #currently, the iou does very little, as it saves all detections
                    intersect:shp.Polygon = shp.intersection(dt_box, tr_box)
                    union = shp.union(dt_box, tr_box)

                    if intersect.is_empty:
                        continue


                    
                    iou = intersect.area / union.area


                    if iou > minimum_iou and true_classes[i][k] in cls_id[j]:

                        
                        valid_idx.append((j,k))
                        global_data[true_classes[i][k]].append(np.asarray(cls_prob[j]))
                    else:
                        global_data[-1].append(np.asarray(cls_prob[j]))
    


                    

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


            if (show_trs != [] and show_gts != [] and visualize or True):
                visualize_multimodal_detections_and_gt(img, boxes, cls_id, cls_prob, true_boxes[i], true_classes[i], block=False)
                visualize_multimodal_detections_and_gt(img, show_trs, show_trcls, show_prob, show_gts, show_gtcls)
               
        #make data NP arrays
        for c in range(-1, self.num_classes):
            if (global_data[c] == []):
                filler = np.zeros(self.num_classes, dtype=np.float64)
                idx = 0 if c < 0 else c
                filler[idx] = 1
                global_data[c].append(filler)
            global_data[c] = np.asarray(global_data[c], dtype=np.float64)

        if (output_file and output_name != ""):
            with open(f'{output_name}.pickle', 'wb') as handle:
                pickle.dump(global_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return global_data
            

            


    def __call__(self, image) -> Any:
        img = tf.cast(image, dtype=tf.float32)
        input_image, ratio = self.prepare_image(img)
        detection = self.model.predict(input_image)


        boxes = detection['boxes'][0]
        delete_idx = []
        cls_prob = detection['cls_prob'][0]

        for i in range(len(boxes)):

            box = boxes[i]
            sBox = shp.box(box[0], box[1], box[0]+box[2], box[1]+ box[3])
            for j in range(len(boxes)):

                if i == j:
                    continue

                box2 = boxes[j]

                sBox2 = shp.box(box2[0], box2[1], box2[0]+box2[2], box2[1]+ box2[3])

                intersect:shp.Polygon = shp.intersection(sBox, sBox2)
                union = shp.union(sBox, sBox2)

                if intersect.is_empty:
                    continue
                
                iou = intersect.area / union.area

                if iou > self.max_iou:
                    if np.argmax(cls_prob[i]) == np.argmax(cls_prob[j]):

                        if (np.max(cls_prob[i]) < np.max(cls_prob[j])):
                            delete_idx.append(i)


        vBox = np.delete(boxes, delete_idx, axis=0)
        vProb = np.delete(cls_prob, delete_idx, axis = 0)


        

        cls_ids = []
 
        for distribs in vProb:

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



        ret = {"boxes":vBox,
               "cls_prob":vProb,
               "cls_ids":cls_ids}
        
        #print(ret)


        return ret
