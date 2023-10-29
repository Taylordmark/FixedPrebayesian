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
The model_trainer file is set up to use coco automatically. You can specify number of epochs with -e and numbers of images samples with -s. It defaults to put the trained model in the folder retinanet ("which may need to be created if it is missing")

# Output Format
The "retina net model" returns a dictionary of two tensors on call "boxes" and "cls_prob", which is a probability vector which should be suitable for a bayesian backend.

Example command: 

python model_trainer.py -j [ANNOTATION PATH] -s train [SAVE PATH]     
