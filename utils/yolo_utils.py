import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

import cv2
import glob

import tensorflow_probability as tfp


def load_images(path:str, resize_size=(640, 640), extension=".jpg"):


    images = []
    if (path.endswith('.mp4')):

        cap = cv2.VideoCapture(path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break

        cap.release()
        return images

    else:


        for pth in glob.glob(f"{path}/*{extension}"):

            im = cv2.imread(pth)

            img = cv2.cvtColor(cv2.resize(im, resize_size), cv2.COLOR_BGR2RGB)
            #img = cv2.resize(im, resize_size)

            if img.shape == resize_size:

                img = cv2.merge([img, img, img])

            images.append(img)
            


        return images


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xywh",
            evaluate_freq=1e9,
        )
        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):

        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)

            print(y_pred)

            y_format = {
                "boxes": y_pred["boxes"],
                "classes": y_pred["cls_idx"],
                "confidence": y_pred["cls_prob"],
            }
            self.metrics.update_state(y_true, y_format)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save_weights(os.path.join(model_dir, f"weights_epoch_{epoch}"))  # Save the model when mAP improves

        return logs
    
# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model