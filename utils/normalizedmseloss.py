from keras.losses import *
import tensorflow as tf


def NormalizedMeanSquaredError(y_true, y_pred):

    tf.print(y_true)

    return mean_squared_error(y_true, y_pred)