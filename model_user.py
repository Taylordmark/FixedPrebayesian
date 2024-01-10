import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import tensorflow_probability as tfp
from utils.coco_dataset_manager import *
from utils.yolo_utils import *
from utils.custom_retinanet import prepare_image
from utils.nonmaxsuppression import *
from utils.negloglikely import nll
from utils.yolov8prob import ProbYolov8Detector
from PIL import Image
import matplotlib.pyplot as plt
import pickle

#import torch

tf.keras.backend.clear_session()
tf.compat.v1.enable_eager_execution()
#torch.cuda.empty_cache()

# Hardcode paths and parameters
checkpoint_path = r"/remote_home/Thesis/Completed_Models/Softmax_mse2"
image_folder = r"/remote_home/Thesis/DataFiles/small_test_videos/BDD_val_b1c9c847-3bda4659"
cls_path = r"/remote_home/Thesis/Prebayesian/class_list_traffic.txt"
download_path = r"/remote_home/Thesis/Prebayesian/download_list_traffic.txt"
loss_function = "mse"  # mse, cce, or pos
nms_layer = 'Softmax'  # Softmax or SoftmaxSum
min_confidence = 0.018
label_smoothing = 0

LEARNING_RATE = 0.0001
GLOBAL_CLIPNORM = 5

# Load the class lists from text; if not specified, it gets all 80 classes
cls_list = None
if cls_path:
    with open(cls_path) as f:
        cls_list = [cls.strip() for cls in f.readlines()]

print(cls_list)

download_list = None
if download_path and download_path != "False":
    with open(download_path) as f:
        download_lines = f.readlines()
        download_list = {line.split(",")[0]: line.split(",")[1].strip() for line in download_lines}

print(download_list)

# The detector will only be the length of the class list
num_classes = 80 if cls_list is None else len(cls_list)

print(num_classes)

# Augmenter and resizing
augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2, bounding_box_format="xywh"),
        keras_cv.layers.JitteredResize(target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xywh"),
    ]
)
resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xywh"
)

# Function to convert dictionary inputs to tuple
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]

# NMS function
nms_fn = DistributionNMS if nms_layer == 'Softmax' else PreSoftSumNMS
detector = ProbYolov8Detector(num_classes, min_confidence=min_confidence, nms_fn=nms_fn)
label_smooth = max(min(label_smoothing, 1), 0)
classification_loss = keras.losses.MeanSquaredError(
    reduction="sum",
)
if loss_function == 'cce':
    classification_loss = keras.losses.CategoricalCrossentropy(
        reduction="sum", from_logits=True, label_smoothing=label_smooth
    )
if loss_function == 'pos':
    classification_loss = keras.losses.Poisson(
        reduction="sum"
    )

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE, global_clipnorm=GLOBAL_CLIPNORM,
)
detector.model.compile(
    optimizer=optimizer, classification_loss=classification_loss, box_loss="ciou", jit_compile=False,
    box_loss_weight=7.5,
    classification_loss_weight=0.5,
)

print("Loading images...")
# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
file_count = len(image_files)
print("Images loaded")

# Load detector Weights
detector.load_weights(checkpoint_path)
print("Detector loaded")

# Define a function to load and preprocess a single image
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(640, 640))
    img_array = img_to_array(img)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32) / 255.0
    return img_tensor

# Assuming detector, cls_list, and image_files are defined in your code
detection_results = {}
prev_max = 0

for frame_number, frame_path in enumerate(image_files):
    checkpoint = round(frame_number / file_count * 100, 0)
    if checkpoint > prev_max:
        print(f"{checkpoint}.2f%")
    prev_max = checkpoint

    frame = load_and_preprocess_image(os.path.join(image_folder, frame_path))

    # Perform object detection
    detections = detector(frame)
    boxes = np.asarray(detections["boxes"])
    cls_prob = np.asarray(detections["cls_prob"])

    saved_boxes = []
    saved_probs = []
    # Get bounding boxes and class names
    for box, prob_list in zip(boxes, cls_prob):
        cur_min = min(prob_list)
        
        probability = max(prob_list)
        
        if probability > cur_min * 1.0015:

            chosen_class = np.argmax(prob_list)
            name = cls_list[chosen_class]

            # Extract coordinates
            ymin, xmin, ymax, xmax = box

            print(f"Index: {np.argmax(prob_list)}, Probability: {probability}")

            saved_boxes.append(box)
            saved_probs.append(prob_list)
    detection_results[frame_number] = {'boxes':saved_boxes, 'probabilities':saved_probs}

# Specify the file path
file_path = 'my_dict.pkl'

# Save the dictionary to a .pkl file
with open(file_path, 'wb') as file:
    pickle.dump(detection_results, file)