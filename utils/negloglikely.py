import tensorflow as tf

from utils.coco_dataset_manager import *
import tensorflow as tf
from utils.yolo_utils import *
from utils.nonmaxsuppression import *
from keras_cv.losses.ciou_loss import CIoULoss


def nll(y_true, y_pred):
    """
    This function should return the negative log-likelihood of each sample
    in y_true given the predicted distribution y_pred. If y_true is of shape 
    [B, E] and y_pred has batch shape [B] and event_shape [E], the output 
    should be a Tensor of shape [B].
    """
    distrib = tfp.distributions.OneHotCategorical(logits=y_pred)
    return -distrib.log_prob(y_true)

    