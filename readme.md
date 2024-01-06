# Overview

This repository is mainly based around "model-main_trainer.py", which trains a YOLOv8 model. More info can be found in the arguments of that script.

# Dataset

The coco dataset is handled mostly automatically, however an annotation file is required as input.

Examples can be found at: https://cocodataset.org/#download
Specifically, download an annotation, from the right column. To prevent an excess of data, when running the model trainer, you can specify the max number of images to download.

Additionally, you can limit the images to only those which include a list of given classes, if desired.

download_list_traffic determines image download classes and counts
class_listt_traffic determines all trained detections classes (may be more than in download_list)

# Setup

Make sure you are using python 3.8.5. I would recommend setting up a virtual environment. Either way, run pip install -r requirements.txt.

# Training
Train using model_trainer-main.py. The dataset is handled automatically, but you can specify how many images you want with -n and you can limit the classes the detector using -l.

# Yolo class list
With the yolo trainer, you can provide the path to a text file of classes to include. These currently must be classes in coco. If you do this, the number of classes will be reduced to the number in the list. Values should be new line delimited. Current classes include people, pets, and traffic objects."

# Model demonstration
Current demonstration runs currectly on "model_demo.ipynb" with a folder of images.

# Output Format
The model outputs a tensor of "boxes", "cls_prob" (probability of each class)

Example command: 

python model_trainer-main.py