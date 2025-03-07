from typing import Any
import tensorflow as tf
from tensorflow import keras
from keras_cv import bounding_box
from keras_cv.backend import ops
from keras_cv.backend.config import multi_backend

import tensorflow_probability as tfp




class PreBayesianNMS(keras.layers.Layer):

    def __init__(
        self,
        bounding_box_format,
        from_logits,
        iou_threshold=0.5,
        confidence_threshold=0.5,
        max_detections=100,
        max_detections_per_class=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.from_logits = from_logits
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.max_detections_per_class = max_detections_per_class
        self.built = True


    def call(
        self, box_prediction, class_prediction, images=None, image_shape=None
    ):
        """Accepts images and raw predictions, and returns bounding box
        predictions.

        Args:
            box_prediction: Dense Tensor of shape [batch, boxes, 4] in the
                `bounding_box_format` specified in the constructor.
            class_prediction: Dense Tensor of shape [batch, boxes, num_classes].
        """

        target_format = "xywh"
        if bounding_box.is_relative(self.bounding_box_format):
            target_format = bounding_box.as_relative(target_format)

        box_prediction = bounding_box.convert_format(
            box_prediction,
            source=self.bounding_box_format,
            target=target_format,
            images=images,
            image_shape=image_shape,
        )

        cls_predictions = class_prediction
        #from [-inf, inf] to [0, 1] with the sum adding up to 1

        print(f"THRESHOLD {self.confidence_threshold}")


        def nms(x):
            """
            Function used to enable nms for tf.map
            """

            box = x[0]
            cls_pred = x[1]

            # Change to regular YOLO by using argmax for probs
            # Create the desired array using tf.where for single-dimensional case:
            cls_pred = tf.where(tf.equal(cls_pred, tf.reduce_max(cls_pred)), 1.0, 0.0)


            #determines indices to keep
            idx = tf.image.non_max_suppression(
                box,
                tf.reduce_max(cls_pred, axis=1),
                self.max_detections,
                self.iou_threshold,
                self.confidence_threshold,
            )


            #only keeps indices in idx
            return tf.gather(box, idx), tf.gather(cls_pred, idx)


        if self.from_logits:
            cls_predictions:tf.Tensor = ops.softmax(cls_predictions)

        nms_box, nms_cls = tf.map_fn(nms, (box_prediction, cls_predictions), dtype=(tf.float32, tf.float32), 
            fn_output_signature=(tf.float32, tf.float32))



        output = {
            "boxes": nms_box,
            "cls_idx": tf.argmax(nms_cls, axis=2),
            "cls_prob": nms_cls,
        }



        return output

class DistributionNMS(keras.layers.Layer):

    def __init__(
        self,
        bounding_box_format,
        from_logits,
        iou_threshold=0.5,
        confidence_threshold=0.5,
        max_detections=1000,
        max_detections_per_class=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.from_logits = from_logits
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.max_detections_per_class = max_detections_per_class
        self.built = True


    def call(
        self, box_prediction, class_prediction, images=None, image_shape=None
    ):
        """Accepts images and raw predictions, and returns bounding box
        predictions.

        Args:
            box_prediction: Dense Tensor of shape [batch, boxes, 4] in the
                `bounding_box_format` specified in the constructor.
            class_prediction: Dense Tensor of shape [batch, boxes, num_classes].
        """

        target_format = "xywh"
        if bounding_box.is_relative(self.bounding_box_format):
            target_format = bounding_box.as_relative(target_format)

        box_prediction = bounding_box.convert_format(
            box_prediction,
            source=self.bounding_box_format,
            target=target_format,
            images=images,
            image_shape=image_shape,
        )

        #from [-inf, inf] to [0, 1] with the sum adding up to 1


        #tf.print(tf.reduce_max(class_prediction, axis=2))

        raw_logits = class_prediction
        
        if self.from_logits:
            class_prediction:tf.Tensor = ops.softmax(class_prediction)


        
        #tf.print(tf.reduce_max(class_confidence))


        def nms(x):
            """
            Function used to enable nms for tf.map
            """

            box = x[0]
            cls_pred = x[1]
            raws = x[2]

            #determines indices to keep
            idx = tf.image.non_max_suppression(
                box,
                tf.reduce_max(cls_pred, axis=1),
                self.max_detections,
                self.iou_threshold,
                self.confidence_threshold,
            )


            #only keeps indices in idx
            nms_box = tf.gather(box, idx)
            nms_cls = tf.gather(cls_pred, idx)
            raw_ret = tf.gather(raws, idx)


            #cls_distrib = self.distrib_fn(logits=nms_cls)

            return nms_box, nms_cls, raw_ret


        nms_box, nms_cls, nms_raw = tf.map_fn(nms, (box_prediction, class_prediction, raw_logits), dtype=(tf.float32, tf.float32, tf.float32), 
            fn_output_signature=(tf.float32, tf.float32, tf.float32))
        

        output = {
            "boxes": nms_box,
            "cls_prob": nms_cls,
            "raw": nms_raw
        }





        return output
    

class PreSoftSumNMS(keras.layers.Layer):

    def __init__(
        self,
        bounding_box_format,
        from_logits,
        iou_threshold=0.5,
        confidence_threshold=0.5,
        max_detections=100,
        max_detections_per_class=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.from_logits = from_logits
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.max_detections_per_class = max_detections_per_class
        self.built = True


    def call(
        self, box_prediction, class_prediction, images=None, image_shape=None
    ):
        """Accepts images and raw predictions, and returns bounding box
        predictions.

        Args:
            box_prediction: Dense Tensor of shape [batch, boxes, 4] in the
                `bounding_box_format` specified in the constructor.
            class_prediction: Dense Tensor of shape [batch, boxes, num_classes].
        """

        target_format = "xywh"
        if bounding_box.is_relative(self.bounding_box_format):
            target_format = bounding_box.as_relative(target_format)

        box_prediction = bounding_box.convert_format(
            box_prediction,
            source=self.bounding_box_format,
            target=target_format,
            images=images,
            image_shape=image_shape,
        )

        cls_predictions = class_prediction
        #from [-inf, inf] to [0, 1] with the sum adding up to 1



        def subtract_min(x):
            cls_pred = x
            my_min = tf.reduce_min(cls_pred)
            cls_pred = tf.add(cls_pred, -my_min)

            return cls_pred


        def nms(x):
            """
            Function used to enable nms for tf.map
            """

            box = x[0]
            cls_pred = x[1]

            #determines indices to keep
            idx = tf.image.non_max_suppression(
                box,
                tf.reduce_max(cls_pred, axis=1),
                self.max_detections,
                self.iou_threshold,
                self.confidence_threshold,
            )


            #only keeps indices in idx
            nms_box = tf.gather(box, idx)
            nms_cls = tf.gather(cls_pred, idx)

            #cls_distrib = self.distrib_fn(logits=nms_cls)

            return nms_box, nms_cls



        nms_box, nms_cls = tf.map_fn(nms, (box_prediction, cls_predictions), dtype=(tf.float32, tf.float32), 
            fn_output_signature=(tf.float32, tf.float32))

        output = {
            "boxes": nms_box,
            "cls_prob": nms_cls,
            "raw":cls_predictions
        }



        return output