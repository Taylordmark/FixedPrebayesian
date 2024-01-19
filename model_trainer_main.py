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
parser.add_argument("--download_path", "-d", type=str, help="Whether to download the dataset images or not", default="download_list_traffic.txt")
parser.add_argument("--batch_size", "-b", type=int, default=16)
parser.add_argument("--epochs", "-e", help="number of epochs", default=500, type=int)
parser.add_argument("--checkpoint_path", "-p", help="path to save checkpoint", default="yolo")
parser.add_argument("--mode", "-m", help="enter train, test, or traintest to do both", default="train", type=str)
parser.add_argument("--max_iou", "-i", help="max iou", default=.125, type=float)
parser.add_argument("--min_confidence", "-c", help="min confidence", default=.5, type=float)
parser.add_argument("--cls_path", "-l", help="path to line seperated class file", default="class_list_traffic.txt", type=str)
parser.add_argument("--loss_function", "-x", help="loss function to use, mse, cce, pos", default="mse", type=str)
parser.add_argument("--label_smoothing", "-o", help="label smoothing for categorical and binary crossentropy losses, ranges from (0, 1)", default=0, type=float)
parser.add_argument("--nms_layer", "-n", help="Which nms layer to use, currently 'Softmax' and 'SoftmaxSum'", type=str, default='Softmax')

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

print(download_list)

#The detector will only be the length of the class list
num_classes = 80 if cls_list is None else len(cls_list)

print(num_classes)

coco_ds = CocoDSManager(args.json_path, args.save_path, download=do_download, 
                        yxyw_percent=False, cls_list=cls_list, download_list=download_list)

train_ds = coco_ds.train_ds
val_ds = coco_ds.val_ds

print("TRAIN DATA LENGTH")
print(len(list(train_ds)))

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

train_ds = train_ds.shuffle(batch_size * 4)
train_ds = train_ds.ragged_batch(batch_size, drop_remainder=True)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.75, 1.3),
    bounding_box_format="xywh",
)

val_ds = val_ds.shuffle(batch_size * 4)
val_ds = val_ds.ragged_batch(batch_size, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]

train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

nms_fn = DistributionNMS if args.nms_layer == 'Softmax' else PreSoftSumNMS

detector = ProbYolov8Detector(num_classes, min_confidence=args.min_confidence, nms_fn=nms_fn)

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

val_loss_history = []  # Create an empty list to store val losses

if "train" in args.mode:
    detector.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            # tf.keras.callbacks.LambdaCallback(
            #     on_epoch_end=lambda epoch, logs: val_loss_history.append(logs["val_loss"])
            # ),
        ],
    )
    # Save the val_loss_history list as a .pkl file
    with open("val_loss_history.pkl", "wb") as f:
        pickle.dump(val_loss_history, f)

if "test" in args.mode:
    detector.load_weights(args.checkpoint_path)

    for sample in coco_ds.train_ds.take(5):
        #try:
            image = tf.cast(sample["images"], dtype=tf.float32)

            detections = detector(image)
            boxes = np.asarray(detections["boxes"])
            cls_prob = np.asarray(detections["cls_prob"])
            cls_id =  np.asarray(detections["cls_ids"])




            cls_name = []
            
            i = 0
            for clses in cls_id:
                names = []
                for cls_n in clses:
                    if (cls_n < 0 or cls_n > len(cls_list)):
                        names.append("unknown")
                    else:
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

            # print(sample["bounding_boxes"]["boxes"])

            # print("VS")
            # print(boxes)


            visualize_multimodal_detections_and_gt(image, boxes, cls_name, correct_prob,
                                        sample["bounding_boxes"]["boxes"], gt_name)
        # except IndexError:
        #     print("NO VALID DETECTIONS")
        #     continue
        #show_frame_no_deep(np.asarray(image), np.asarray(detections["boxes"][0]), 2000)
