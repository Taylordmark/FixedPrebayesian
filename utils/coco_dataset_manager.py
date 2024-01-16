import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


from PIL import Image

from pycocotools.coco import COCO

import requests


import cv2
from utils.retina_net.retina_label_encoder import *

class CocoDSManager:
    """
    Class to manage the coco dataset and allow for smaller subsets

    annotation_pth:str path to a coco annotation json file, which can be downloaded here: https://cocodataset.org/#download
    save_pth:str directory to save images
    cls_list: which classes to download images with, leave blank to get all
    """
    @tf.autograph.experimental.do_not_convert
    def __init__(self, annotation_pth:str, save_pth:str, max_samples:int=60, test_split_denominator:int = 5, 
                 cls_list:list=None, download_list:list=None, download=True, resize_size=(640, 640), yxyw_percent=True) -> None:
        self.ann_pth = annotation_pth
        self.save_pth = save_pth
        self.cls_list = cls_list
        self.download_list = download_list
        self.split = test_split_denominator

        self.coco = COCO(self.ann_pth)

        self.yxyw_percent = yxyw_percent

        # instantiate COCO specifying the annotations json path

        coco = self.coco
        key_list = list(coco.cats.keys())



        # Specify a list of category names of interest
        if self.download_list is not None:

            cat_list = self.cls_list if self.download_list is None else self.download_list.keys()
            catIds = coco.getCatIds(catNms=cat_list)

            clsIds = coco.getCatIds(catNms=self.cls_list)


            

            imgIds = []

            for cat in catIds:  
                tempIds = coco.getImgIds(catIds=[cat])

                key = coco.cats[cat]


                i = 0
                for ids in tempIds:
                    if i < int(self.download_list[key['name']]) and ids not in imgIds:
                        imgIds.append(ids)
                        i += 1  


            imgIds = list(set(imgIds))

            key_list = list(clsIds)
        else:
            imgIds = coco.getImgIds()
        # Get the corresponding image ids and images using loadImgs

        img_to_load = []
        labels = coco.loadAnns(coco.getAnnIds(imgIds))

        bboxes = []
        cls_ids = []

        i = 0
        j = 0
        split_list = []

        images = []



        for label in labels:

            if (j >= len(imgIds)):
                break


            if (label["category_id"] not in key_list):
                continue
            
            img = coco.loadImgs([label["image_id"]])[0]



            size = (img['width'], img['height'])
            

            if imgIds[j] == label["image_id"]:
                split_list.append(i)
                images.append(img)
                img_to_load.append(imgIds[j])
                j += 1
            i += 1

            resized = resize_xywh(label["bbox"], size, resize_size)
            if self.yxyw_percent:
                resized = xywh_to_yxyx_percent(resized, resize_size)
            bboxes.append(resized)
            #bboxes.append(resize_xywh(label["bbox"], size, resize_size))

            cls_ids.append(key_list.index(label["category_id"]))



        #downloads images to disk 
        images = coco.loadImgs(img_to_load)

        print(f"LOADING {len(img_to_load)} IMAGES")
        for im in images:

            name = os.path.join(self.save_pth, im['file_name'])

            if not os.path.exists(self.save_pth):
                os.makedirs(self.save_pth)
            
            if not os.path.isfile(name):
                img_data = requests.get(im['coco_url']).content
                with open(name, 'wb') as handler:
                    handler.write(img_data)



        split_list.append(len(bboxes))

        images = self.load_images(self.save_pth, img_to_load, resize_size)

        box_tens = tf.RaggedTensor.from_row_splits(bboxes, split_list)
        cls_tens = tf.RaggedTensor.from_row_splits(cls_ids, split_list)

        full_ds = tf.data.Dataset.from_tensor_slices(
            {"images":images,
             "bounding_boxes": {
             "boxes": box_tens,
             "classes": cls_tens}
            })
        
        val_test_ds = full_ds.enumerate() \
                    .filter(lambda x,y: x % self.split == 0) \
                    .map(lambda x,y: y)
        
        
        self.train_ds = full_ds.enumerate() \
                    .filter(lambda x,y: x % self.split != 0) \
                    .map(lambda x,y: y)
        
        self.val_ds = val_test_ds.enumerate() \
                    .filter(lambda x,y: x % 2 == 0) \
                    .map(lambda x,y: y)
        
        
        self.test_ds = val_test_ds.enumerate() \
                    .filter(lambda x,y: x % 2 != 0) \
                    .map(lambda x,y: y)

        self.key_list = key_list
        

    def load_images(self, path:str, ids, resize_size=(640, 640), extension=".jpg"):

        images = []


        for id in ids:
            f = path+"\\"+(str(id).zfill(12))+extension
            #name, _ = f.replace(path, "").replace("\\", "").lstrip("0").split(".")

            img = cv2.resize(np.asarray(Image.open(f)), resize_size)

            if img.shape == resize_size:

                img = cv2.merge([img, img, img])

            images.append(img)
            


        return images

def resize_xywh(xywh, old_size, new_size):
    ratio = (old_size[0]/new_size[0], old_size[1]/new_size[1])

    return [xywh[0] / ratio[0], xywh[1] / ratio[1], xywh[2] / ratio[0], xywh[3]/ ratio[1]]


def xywh_to_yxyx_percent(xywh, img_size):
    return [xywh[1]/ img_size[1], xywh[0]/img_size[0], (xywh[1]+xywh[3])/img_size[1], (xywh[0]+xywh[2])/img_size[0]]

def yxyx_percent_to_xywh(xyxy, img_size):

    return [xyxy[1]*img_size[1], xyxy[0]*img_size[0], (xyxy[3] -xyxy[1])*img_size[1], (xyxy[2] -xyxy[0])*img_size[0]]


def format_dataset(train_dataset, autotune, label_encoder, batch_size=4, ignore_errors=True):
    
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)

    train_dataset = train_dataset.shuffle(8 * batch_size)
    #train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    if ignore_errors:
        train_dataset = tf.data.Dataset.ignore_errors(train_dataset)
    train_dataset = train_dataset.prefetch(autotune)

    return train_dataset


        



                    
