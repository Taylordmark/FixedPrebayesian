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



tf.keras.backend.clear_session()

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

parser.add_argument("--json_path", "-j", type=str, help="Path of the coco annotation used to download the dataset", default="annotations/instances_train2017.json")
parser.add_argument("--save_path", "-s", type=str, help="Path to save \ load the downloaded dataset", default="train")
parser.add_argument("--download", "-d", type=str, help="Whether to download the dataset images or not", default="True")
parser.add_argument("--batch_size", "-b", type=int, default=4)
parser.add_argument("--epochs", "-e", help="number of epochs", default=50, type=int)
parser.add_argument("--num_imgs", "-n", help="number of images", default=50, type=int)
parser.add_argument("--checkpoint_path", "-p", help="path to save checkpoint", default="yolo")
parser.add_argument("--mode", "-m", help="enter train, test, or traintest to do both", default="traintest", type=str)
parser.add_argument("--max_iou", "-i", help="max iou", default=.2, type=float)
parser.add_argument("--min_confidence", "-c", help="min confidence", default=.01, type=float)



args = parser.parse_args()

model_dir = args.checkpoint_path

num_classes = 80
batch_size = args.batch_size

#TODO UNCOMMENT THESE LINES TO USE DEFAULT LEARNING RATES

# learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
#learning_rate_boundaries = [125, 250, 500, 240000, 360000]


b_mod = int(args.num_imgs/batch_size)


#TODO COMMENT THE NEXT 2 LINES IF USING DEFAULT RATES
learning_rate_boundaries = [5*b_mod, 10*b_mod, 25*b_mod, 50*b_mod, 100*b_mod]
learning_rates = [.0005, .001, 0.005, 0.001, 0.0005, 0.00025]

learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

coco_ds = CocoDSManager(args.json_path, args.save_path, max_samples=args.num_imgs, download=args.download == "True", yxyw_percent=False)


train_ds = coco_ds.train_ds
val_ds = coco_ds.val_ds


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

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
)

model = keras_cv.models.YOLOV8Detector(
    num_classes=80,
    bounding_box_format="xywh",
    backbone=backbone,
    fpn_depth=1,
)

LEARNING_RATE = 0.0001
GLOBAL_CLIPNORM = 10.0

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

model.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
)

if "train" in args.mode:

    model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=[tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),],
    )


if "test" in args.mode:
#for i in range(args.epochs):
    latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_path)
    
    #print(latest_checkpoint)
    model.load_weights(latest_checkpoint).expect_partial()

    #model.load_weights(r"retinanet\weights_epoch_"+str(124)).expect_partial()

    # print(f"WEIGHTS {124}")


    for sample in coco_ds.train_ds.take(10):


        image = tf.cast(sample["images"], dtype=tf.float32)

    
        input_image, ratio = prepare_image(image)
        detections = model.predict(input_image)

        print(detections)

        boxes = np.asarray(detections["boxes"])

        cls_prob = np.asarray(detections["confidence"])

        # print(np.max(cls_prob))
        # print(np.sum(cls_prob))

        cls_id = np.asarray(detections["classes"])

        key_list = list(coco_ds.coco.cats.keys())

        #print(cls_id)
        
        cls_name = [coco_ds.coco.cats[key_list[x]] for x in cls_id[0]]

                             
        print(sample["bounding_boxes"]["boxes"][:3])


        # visualize_dataset(image, sample["bounding_boxes"]["boxes"][:3], sample["bounding_boxes"]["classes"][:3])
        # visualize_detections(image, boxes[0], cls_id[0], cls_prob[0])

        visualize_detections_and_gt(image, boxes[0], cls_id[0], cls_prob[0],
                                    sample["bounding_boxes"]["boxes"][:3], sample["bounding_boxes"]["classes"][:3])
        #show_frame_no_deep(np.asarray(image), np.asarray(detections["boxes"][0]), 2000)
