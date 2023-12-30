import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

def visualize_detections_multimodal_classes(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = ""
        for i in range(len(_cls)):
            text += "{}: {:.2f} - ".format(_cls[i], score[i])
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax

def visualize_multimodal_detections_and_gt(
    image, boxes, classes, scores,  
    
    boxes_gt, classes_gt, figsize=(7, 7),
    
    linewidth=1, color=[0, 0, 1],

    linewidth_gt=1, color_gt=[0, 1, 0]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = ""
        for i in range(len(_cls)):
            text += "{}: {:.2f} - ".format(_cls[i], score[i])
            
        x1, y1, w, h = box
        #w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )

    for box, _cls in zip(boxes_gt, classes_gt):
        text = "{}".format(_cls)
        x1, y1, w, h = box
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color_gt, linewidth=linewidth_gt
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax

