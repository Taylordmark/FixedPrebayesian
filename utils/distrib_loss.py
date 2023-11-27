import tensorflow as tf

from utils.coco_dataset_manager import *
import tensorflow as tf
from utils.yolo_utils import *
from utils.nonmaxsuppression import *
from keras_cv.losses.ciou_loss import CIoULoss


class DistributionLoss(tf.losses.Loss):
        
    def __init__(self, num_classes, distrib_fn, raw_multiplier=1, reduction="auto") -> None:
        super().__init__(reduction=reduction)
        self.num_classes = num_classes
        self.distrib_fn = distrib_fn
        self.raw_multi = raw_multiplier
        #self.box_loss_fn = CIoULoss(bounding_box_format=bb_format, reduction=reduction)

    def call(self, y_pred, y_true):
            y_pred = tf.cast(y_pred, dtype=tf.float32)

            cls_distrib = self.distrib_fn(logits=y_pred)

            cls_loss = self.nll(y_true, cls_distrib)

            return cls_loss

    def nll(self, y_true, y_pred):
        """
        This function should return the negative log-likelihood of each sample
        in y_true given the predicted distribution y_pred. If y_true is of shape 
        [B, E] and y_pred has batch shape [B] and event_shape [E], the output 
        should be a Tensor of shape [B].
        """

        return -y_pred.log_prob(y_true)*self.raw_multi

    