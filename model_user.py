import argparse
import tensorflow as tf
import numpy as np
from utils.coco_dataset_manager import *
import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow import keras
import keras_cv
from utils.yolo_utils import *
from utils.custom_retinanet import prepare_image
from utils.nonmaxsuppression import *
from utils.negloglikely import nll
import tensorflow_probability as tfp
from utils.yolov8prob import ProbYolov8Detector
from utils.visualization_functions import visualize_multimodal_detections_and_gt

from keras_cv.losses.ciou_loss import CIoULoss
import pickle

from utils.normalizedmseloss import NormalizedMeanSquaredError

tf.keras.backend.clear_session()
tf.compat.v1.enable_eager_execution()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs," , len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser(description="Model Trainer")
parser.add_argument("--json_path", "-j", type=str, help="Path of the coco annotation used to download the dataset", default=r"C:\Users\keela\Documents\annotations\instances_train2017.json")
parser.add_argument("--save_path", "-s", type=str, help="Path to save \ load the downloaded dataset", default=r"C:\Users\keela\Documents\train")
parser.add_argument("--download_path", "-d", type=str, help="Whether to download the dataset images or not", default="download_list_traffic_small.txt")
parser.add_argument("--batch_size", "-b", type=int, default=16)
parser.add_argument("--epochs", "-e", help="number of epochs", default=500, type=int)
parser.add_argument("--checkpoint_path", "-p", help="path to save checkpoint", default=r"C:\Users\keela\Documents\Models\LastMinuteRuns\Small_MLE")
parser.add_argument("--mode", "-m", help="enter train, test, or traintest to do both", default="test", type=str)
parser.add_argument("--max_iou", "-i", help="max iou", default=.125, type=float)
parser.add_argument("--min_confidence", "-c", help="min confidence", default=.13, type=float)
parser.add_argument("--cls_path", "-l", help="path to line seperated class file", default="class_list_traffic.txt", type=str)
parser.add_argument("--loss_function", "-x", help="loss function to use, mse, cce, pos", default="mse", type=str)
parser.add_argument("--label_smoothing", "-o", help="label smoothing for categorical and binary crossentropy losses, ranges from (0, 1)", default=0, type=float)
parser.add_argument("--nms_layer", "-n", help="Which nms layer to use, currently 'Softmax' and 'SoftmaxSum'", type=str, default='Softmax')
parser.add_argument("--backbone_size", "-z", help="what size of yolo backbone to use, defaults to s, l also possible", default="s")

test_image_folder = r"C:\Users\keela\Documents\Videos\train1\000f8d37-d4c09a0f"

args = parser.parse_args()
model_dir = args.checkpoint_path
batch_size = args.batch_size
do_download = args.download_path != "False"

LEARNING_RATE = 0.00015
GLOBAL_CLIPNORM = 5


#Load the class lists from text, if not specified, it gets all 80 classes
if (args.cls_path == ""):
    cls_list = None
else:
    with open(args.cls_path) as f:
        cls_list = f.readlines()
        cls_list = [cls.replace("\n", "") for cls in cls_list]
print(cls_list)

if (args.download_path == "" or args.download_path == "False"):
    download_list = None
else:
    with open(args.download_path) as f:
        download_lines = f.readlines()
        download_list = {}
        for line in download_lines:
            split = line.replace("\n", "").split(",")
            download_list[split[0]]=split[1]


#The detector will only be the length of the class list
num_classes = 80 if cls_list is None else len(cls_list)

augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        keras_cv.layers.RandomShear(
            x_factor=0.2, y_factor=0.2, bounding_box_format="xywh"
        ),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xywh"
        ),
    ]
)


resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.75, 1.3),
    bounding_box_format="xywh",
)


def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


nms_fn = DistributionNMS if args.nms_layer == 'Softmax' else PreSoftSumNMS

backbone_nm = f"yolo_v8_{args.backbone_size}_backbone_coco"
detector = ProbYolov8Detector(num_classes, min_confidence=args.min_confidence, nms_fn=nms_fn, backbone_name=backbone_nm, min_prob_diff=.015)

#distrib_loss = tfp.experimental.nn.losses.neg

label_smooth = max(min(args.label_smoothing, 1), 0)


classification_loss = NormalizedMeanSquaredError(reduction="sum")
    
if args.loss_function == 'cce':
    classification_loss = keras.losses.CategoricalCrossentropy(
        reduction="sum",
        from_logits=True,
        label_smoothing=label_smooth
    )
if args.loss_function == 'sce': #This is likely wrong, since we are using one hot encoded labels
    classification_loss = keras.losses.SparseCategoricalCrossentropy (
        reduction="sum",
        from_logits=True
    )
if args.loss_function == 'mse':
    classification_loss = keras.losses.MeanSquaredError(
        reduction="sum"
    )
if args.loss_function == 'mle':
    classification_loss = keras.losses.MeanSquaredLogarithmicError(
        reduction="sum"
    )
if args.loss_function == 'nme':
    classification_loss = NormalizedMeanSquaredError(
        reduction="sum"
        )
if args.loss_function == 'pos':
    classification_loss = keras.losses.Poisson (
        reduction="sum"
    )

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

box_loss = CIoULoss(bounding_box_format="xywh", reduction="sum")

detector.model.compile(
    optimizer=optimizer, classification_loss=classification_loss, box_loss=box_loss, jit_compile=False,
    box_loss_weight=10,
    classification_loss_weight=5,
)


detector.load_weights(args.checkpoint_path)


# Get a list of image files in the folder
image_files = [f for f in sorted(os.listdir(test_image_folder)) if f.lower().endswith(".jpg")]
total_files = len(image_files)
prev_checkpoint = 0

detections_dict = {}
for frame_num, image_file in enumerate(image_files):
    progress = round(frame_num / total_files, 1)
    if progress > prev_checkpoint:
        prev_checkpoint = progress
        print(f"{prev_checkpoint:.2f}")
    # Construct the full path to the image
    image_path = os.path.join(test_image_folder, image_file)
    tensor_image = tf.keras.utils.load_img(image_path)
    reshaped_tensor = tf.image.resize(tensor_image, size=(640, 640))

    # Run the detector on the image
    detections = detector(reshaped_tensor)
    boxes = np.asarray(detections["boxes"])
    cls_prob = np.asarray(detections["cls_prob"])

    detections_dict[frame_num] = {'boxes': boxes, 'probabilities': cls_prob}

pickle_path = os.path.join(args.checkpoint_path, '000f8d37-d4c09a0f_initial_detections.pkl')
with open(pickle_path, 'wb') as pkl:
    pickle.dump(detections_dict, pkl)
    
      