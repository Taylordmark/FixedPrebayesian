# Overview

This repository is mainly based around "model_trainer.py", which trains a retina net model. More info can be found in the arguments of that script.

It is modified from: https://keras.io/examples/vision/retinanet/

# Dataset

The coco dataset is handled mostly automatically, however an annotation file is required as input.


Examples can be found at: https://cocodataset.org/#download
Specifically, download an annotation, from the right column. To prevent an excess of data, when running the model trainer, you can specify the max number of images to download.

Additionally, you can limit the images to only those which include a list of given classes, if desired.

# Setup

Make sure you are using python 3.8.5. I would recommend setting up a virtual environment. Either way, run pip install -r requirements.txt.

# Training
There are two different model trainers, one for retina net and one for yolo, of which yolo is labeled accordingly. The dataset is handled automatically, but you can specify how many images you want with -n and you can limit the classes the detector using -l. Training on server is being conducted using "edited_yolo_trainer.py".

# Yolo class list
With the yolo trainer, you can provide the path to a text file of classes to include. These currently must be classes in coco. If you do this, the number of classes will be reduced to the number in the list. Values should be new line delimited. Current classes include people, pets, and traffic objects."

# Model demonstration
Current demonstration runs currectly on "yolo_demonstrator.ipynb" with coco images only. Further work to be done with making the predictor applicable to general .jpg or .png images.

# Output Format
Both models output a tensor of "boxes", "cls_prob" (probability of each class), and "cls_idx" (index of highest probability class for each prediction)

Example command: 

python model_trainer-yolo.py -e 50 -s [train-pth] -j [path-to-coco-json] -n 1000 -c .45 -l yolo-cls-list.txt
