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
checkpoint_path = r"/remote_home/Thesis/Models/cce_ls"
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

download_list = None
if download_path and download_path != "False":
    with open(download_path) as f:
        download_lines = f.readlines()
        download_list = {line.split(",")[0]: line.split(",")[1].strip() for line in download_lines}

# The detector will only be the length of the class list
num_classes = 80 if cls_list is None else len(cls_list)

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
    classification_loss_weight=5,
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

detector.load_weights(args.checkpoint_path)

for sample in coco_ds.train_ds.take(5):
    image = tf.cast(sample["images"], dtype=tf.float32)

    detections = detector(image)
    boxes = np.asarray(detections["boxes"])
    cls_prob = np.asarray(detections["cls_prob"])
    cls_id = []

    for distribs in cls_prob:
        i = 0
        ids = []
        min = np.min(distribs)
        for prob in distribs:
            if prob > min+.005:
                ids.append(i)
            i +=1
        cls_id.append(ids)
    

    cls_name = []
    for clses in cls_id:
        names = []
        for cls_n in clses:
            names.append(cls_list[cls_n])
        cls_name.append(names)

    key_list = coco_ds.key_list
    
    print(cls_prob)

    correct_prob = []
    for i in range(len(cls_prob)):

        probs = []
        for ids in cls_id[i]:
            probs.append(cls_prob[i][ids])
        correct_prob.append(probs)
    
    gt_name = [coco_ds.coco.cats[key_list[int(x)]]['name'] for x in np.asarray(sample["bounding_boxes"]["classes"])]
            
    # visualize_dataset(image, sample["bounding_boxes"]["boxes"][:3], sample["bounding_boxes"]["classes"][:3])
    # visualize_detections(image, boxes[0], cls_id[0], cls_prob[0])

    print(sample["bounding_boxes"]["boxes"])

    print("VS")
    print(boxes)

    visualize_multimodal_detections_and_gt(image, boxes, cls_name, correct_prob,
                                sample["bounding_boxes"]["boxes"], gt_name)


# Checks if any dict values are > 30, returns True or False

def all_values_length_gt_0(data_dict):
    """
    Checks if all values in the dictionary are np arrays with no elements between 0 and 30 (exclusive).

    Args:
        data_dict: A dictionary where values are np arrays.

    Returns:
        True if all values have no elements > 30, False otherwise.
    """
    for value in data_dict.values():
        if len(value) > 30:
            return False
        else:
            return True


# Calculates and returns all global distribution parameters
def global_parameterize(data_dict):
    """
    Recalculates and returns all global distribution parameters.

    Args:
        data_dict: A dictionary where values are np arrays.

    Returns:
        A list of lists containing distribution parameters for each column.
    """
    # Initialize an empty list of lists to store parameters
    distribution_parameters = []

    # For each class in the data dictionary
    for k, history in data_dict.items():
        # Initialize an empty list to store parameters for the current class
        col_parameters = []

        # Iterate through each column of the data matrix
        for col in range(len(history[0])):
            # Extract data for the current column
            data = history[:, col]

            # Calculate distribution statistics
            a, b, loc, scale = beta.fit(data)

            # Store parameters for the current column
            col_parameters.append([a, b, loc, scale])

        # Append parameters for the current class to the main list
        distribution_parameters.append(col_parameters)

    return distribution_parameters

# Recalculates and returns a local distribution parameter
def local_parameterize(history):
    """
    Recalculates and returns distribution parameters for a single history.

    Args:
        history: A numpy array representing a single history.

    Returns:
        A list containing distribution parameters for each column.
    """
    # Initialize an empty list to store parameters
    col_parameters = []

    # Iterate through each column of the history
    for col in range(len(history[0])):
        # Extract data for the current column
        data = history[:, col]

        # Calculate distribution statistics
        a, b, loc, scale = beta.fit(data)

        # Store parameters for the current column
        col_parameters.append([a, b, loc, scale])

    return col_parameters


print("Filling globals")

while all_values_length_gt_0(global_data):
    for frame, result in loaded_frames_detections.items():
        for box_num, box in enumerate(result['probabilities']):
            fake_prediction = result['classes'][box_num]
            global_data[fake_prediction] = np.vstack((global_data[fake_prediction], box))

print("Globals filled")