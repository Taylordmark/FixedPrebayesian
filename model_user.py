import tensorflow as tf
from tensorflow.keras.models import load_model

# Import the custom layers, models, and loss function here
import keras_cv.models
from utils.custom_retinanet import prepare_image
from utils.nonmaxsuppression import PreBayesianNMS
from utils.losses import CIoULoss  # Adjust the import statement based on the actual location

# Set up a custom object scope to register the custom layers, models, and loss function
with tf.keras.utils.custom_object_scope({
    "YOLOV8Backbone": keras_cv.models.YOLOV8Backbone,
    "YOLOV8Detector": keras_cv.models.YOLOV8Detector,
    "PreBayesianNMS": PreBayesianNMS,
    "CIoULoss": CIoULoss,  # Register the custom loss function
    # Add any other custom layers, models, or loss functions you are using
}):
    loaded_model = load_model("latest_model.h5")



image = preprocess_image("/remote_home/Thesis/BDD_Files/traffic/frame_0000.png")
input_image, ratio = prepare_image(image)
detections = loaded_model.predict(input_image)

print(detections)