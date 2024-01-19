import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
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

checkpoint_path = 'mean_squared_error'
min_confidence = 0.55
cls_path = 'custom-cls.txt'
download_path = 'download_list.txt'
save_path = 'train'
json_path = r'C:\Users\nloftus\Documents\Datasets\coco_2017_annotations\instances_train2017.json'



cls_list = None
if cls_path:
    with open(cls_path) as f:
        cls_list = [cls.strip() for cls in f.readlines()]

download_list = None
if download_path and download_path != "False":
    with open(download_path) as f:
        download_lines = f.readlines()
        download_list = {line.split(",")[0]: line.split(",")[1].strip() for line in download_lines}

num_classes = 80 if cls_list is None else len(cls_list)

coco_ds = CocoDSManager(json_path, save_path, download=True, 
                        yxyw_percent=False, cls_list=cls_list, download_list=download_list)

nms_fn = DistributionNMS 
detector = ProbYolov8Detector(num_classes, min_confidence=min_confidence, nms_fn=nms_fn)


truth_data = []
images = []
for sample in coco_ds.train_ds.take(5):

    image = tf.cast(sample["images"], dtype=tf.float32)
    images.append(image)
    truth_data.append(sample['bounding_boxes'])

detector.generate_global_data(images, truth_data)