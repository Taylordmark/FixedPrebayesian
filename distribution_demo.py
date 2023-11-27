import argparse

import tensorflow as tf
import numpy as np

from utils.coco_dataset_manager import *

import tensorflow as tf

import keras_cv

from utils.yolo_utils import *

from utils.custom_retinanet import prepare_image

from utils.nonmaxsuppression import *

from utils.distrib_loss import *

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

parser.add_argument("--image_dir", "-d", help="path to test image directory", default="train")
parser.add_argument("--checkpoint_path", "-p", help="path to load checkpoint", default="best_weights")
parser.add_argument("--max_iou", "-i", help="max iou", default=.2, type=float)
parser.add_argument("--min_confidence", "-c", help="min confidence", default=.1, type=float)
parser.add_argument("--cls_path", "-l", help="path to line seperated class file", default="", type=str)


args = parser.parse_args()


#Load the class lists from text, if not specified, it gets all 80 classes
if (args.cls_path == ""):
    cls_list = None
else:
    with open(args.cls_path) as f:
        cls_list = f.readlines()
        cls_list = [cls.replace("\n", "") for cls in cls_list]
num_classes = 80 if cls_list is None else len(cls_list)




distrib_fn = tfp.distributions.Binomial
nms = DistributionNMS("xywh", True, distrib_fn, confidence_threshold=args.min_confidence)

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
)

model = keras_cv.models.YOLOV8Detector(
    num_classes= num_classes,
    bounding_box_format="xywh",
    backbone=backbone,
    fpn_depth=2,
    prediction_decoder=nms
)

for layer in model.layers:
    if "conv" in layer.name:
       layer = tfp.layers.Convolution2DFlipout(
           filters=layer.filters,
           kernel_size=layer.kernel_size,
           strides=layer.strides,
           padding=layer.padding,
           data_format=layer.data_format,
           dilation_rate=layer.dilation_rate,
           activation=layer.activation,
       )

images = load_images(args.image_dir)


latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_path)

model.load_weights(latest_checkpoint).expect_partial()

ds = tf.data.Dataset.from_tensor_slices(images)


for img in ds:
    #try:
        image = tf.cast(img, dtype=tf.float32)

        input_image, ratio = prepare_image(image)
        detections = model.predict(input_image)

        nms_cls = detections["cls_prob"]

        cls_prob = []
        cls_id = []



        for cls in nms_cls[0]:

            
            dist = tfp.distributions.Categorical(logits=cls)

            soft = tf.nn.softmax(logits=cls)


            n = 1e6
            mean = tf.cast(
                tf.histogram_fixed_width(
                dist.sample(int(n)),
                [0, num_classes],
                nbins=num_classes),
                dtype=tf.float32) / (n)
            
            if (np.max(mean) < args.min_confidence):
                continue
            #print(mean)
            #print(f"real {np.argmax(cls)} vs {np.argmax(mean)} sum is {np.sum(mean)} max is {np.max(mean)} softmax is {np.max(soft.numpy())}")


            cls_prob.append(mean)
            cls_id.append(np.argmax(mean))

        #print(cls_prob)


        boxes = np.asarray(detections["boxes"][0])


        
        correct_prob = []
        for i in range(len(cls_prob)):
            correct_prob.append(cls_prob[i][cls_id[i]])
        

        #print(np.max(cls_prob))

        cls_name = [cls_list[x] for x in cls_id]


        visualize_detections(image, boxes, cls_name, correct_prob)
    #except IndexError:
     #   print("NO VALID DETECTIONS")