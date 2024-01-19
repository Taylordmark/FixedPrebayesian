from keras.losses import *
from keras.src.utils.losses_utils import ReductionV2
import tensorflow as tf




class NormalizedMeanSquaredError(Loss):

    def __init__(self, reduction=ReductionV2.AUTO, activation="softmax",name=None):

        self.activation_fn = tf.nn.softmax if activation=="softmax" else tf.nn.sigmoid

        super().__init__(reduction, name)


    def call(self, y_true, y_pred):


        y_p_soft = self.activation_fn(y_pred)
        
        return mean_squared_error(y_true, y_p_soft)