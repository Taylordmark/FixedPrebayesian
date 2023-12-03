
# Python program to read
# json file
  
  
import json
# import scalabel.tools.edit_labels as el
# import scalabel.label.transforms as tr
import glob
import numpy as np
import keras
import pprint
#import plaidml.keras
#from plaidml.keras import backend
from PIL import Image
import utils.bddclassmanager as bdc
#import kerasresnetutils as krnu

  
def get_image_list_from_dir(dir:str, in_filetype:str="jpg", out_format:str="tensor", max_num=-1):

    image_list = []
    print("outputting " + dir+'/*.'+in_filetype + " as " + out_format)
    count = 0
    for filename in glob.glob(dir+'/*.'+in_filetype): #defaultstojpg
        count += 1
        if (max_num > 0 and count > max_num):
            break
        im=Image.open(filename)
        if (out_format == "img"):
            image_list.append(im)
        else:
            image_list.append(np.asarray(im))
        

    print("appended all images")

    if (out_format == "img"):
        print("returning img")
        return image_list
    else:
        if (out_format == "tensor"):
            #image_list = backend.constant(image_list)
            print("returning tensor")
            return image_list
        image_list = np.asarray(image_list)
        print("returning np")
        return image_list
    
    
# def get_scalabels_from_dir(dir:str, images, out_format:str="np"):
#     print("outputting " + dir+'/*.json')
#     class_list = []
#     label_list = []
#     name_list = []

#     max_depth = 0

#     for filename in glob.glob(dir+'/*.json'):

#         json_obj = el.read_input(filename)

#         for entry in json_obj:
#             box_per_entry = []
#             lab_box_arr = []
#             name = entry['name']

#             for box in entry['labels']:
#                 box_arr = []
#                 x1 = box['box2d']['x1']
#                 x2 = box['box2d']['x2']
#                 y1 = box['box2d']['y1']
#                 y2 = box['box2d']['y2']

#                 box_arr.append(x1)
#                 box_arr.append(y1)
#                 box_arr.append(x2-x1)
#                 box_arr.append(y2-y1)
#                 if out_format == "np":
#                     box_arr = np.asarray(box_arr)
#                 #print(box_arr.shape)
#                 box_per_entry.append(box_arr)

#                 #todo add class encoding
#                 box_str = box["category"]
#                 lab_box_arr.append(bdd_pascal_voc_class_list.index(box_str))

#             if out_format == "np":
#                 box_per_entry = np.asarray(box_per_entry)
#                 current_depth = box_per_entry.shape[0]
#                 #print(current_depth)
#                 if (current_depth > max_depth):
#                     max_depth = current_depth
#             label_list.append(box_per_entry)
#             class_list.append(lab_box_arr)
#             name_list.append(name)

#     if out_format == "np":
#         label_list = np.asarray(label_list)
#         class_list = np.asarray(class_list)

#     dict_return = {
#         "boxes": label_list,
#         "classes": class_list,
#         "name": name_list,
#     }
    
#     return dict_return

# def load_labels_from_single_file(filename:str):
#     return_list = []
#     json_obj = el.read_input(filename)
#     for frame in json_obj:
#         return_list.append(frame)

#     return return_list

# def load_labels_by_frame(dir:str):
#     return_list = []
#     for filename in glob.glob(dir+'/*.json'):
#         json_obj = el.read_input(filename)
#         for frame in json_obj:
#             return_list.append(frame)

#     return return_list

# def load_labels_per_entry_seperate(dir:str):
#     return_list = {}
#     for filename in glob.glob(dir+'/*.json'):
#         entry = []
#         json_obj = el.read_input(filename)
#         for frame in json_obj:
#             entry.append(frame)

#         name_l = filename.replace(dir, '').replace('.json', "").split("_")[1:]

#         name = ""
#         for elem in name_l:
#             name += elem
#         return_list[name] = (entry)

#     return return_list

def mot_to_sclb(file_path:str):
    output = TestDirectoryToScalable(file_path)
    file1 = open(file_path,"r")
    lines = file1.readlines()


    entries = {}

    for line in lines:
        spt = line.split(",")
        
        #MOT format lacks class
        fdx = int(spt[0])

        if fdx not in entries.keys():

            entries[fdx] = {
                "t_id": [],
                "box": [],
                "conf": [],
                "cls": []
            }

        t_id = spt[1]
        box = [float(spt[2]), float(spt[3]), (float(spt[2])+float(spt[4])), (float(spt[3])+float(spt[5]))]

        conf = spt[6]
        cls = spt[7]

        entries[fdx]["box"].append(box)
        entries[fdx]["conf"].append(conf)
        entries[fdx]["cls"].append(cls)
        entries[fdx]["t_id"].append(t_id)


    i = 0
    for key in entries:

        entry = entries[key]
        output.add_frame_entry(i, entry["box"], entry["conf"], entry["cls"], entry["t_id"])

        i += 1


    output.output_scalabel_detections(file_path)


"""
Class for outputting detections to scalabel format, for berkeley Deep Drive evaluation
"""
class TestDirectoryToScalable:

    """
    dir_name:str = path to directory
    file_type:str = type of image
    """
    def __init__(self, dir_name:str, file_type:str=".jpg", cls_manager:bdc.BDDCategory = bdc.YOLO_CATEGORIES):
        self.dir_name = dir_name
        self.file_type=".jpg"
        self.data = []
        self.cls_manager:bdc.BDDCategory = cls_manager
        self.id = 0
        self.track_continuity = {}


    """
    Adds a frame to list of entries

    frame_index:int = number of the frame in the video
    xywh:list[list[float]] = list of 4 elements list storing the x and y (top left corner),
    as well as the width and height of each box in the frame
    confidence:list[float] = list of confidences for each detection
    classes:list[float] = index of the class for each detections

    returns: none
    """
    def add_frame_entry(self, frame_index:int, xyxy:list, confidence:list, classes:list, ids:list):
        frame_entry = {}
        str_index = str(frame_index+1).zfill(7)
        frame_entry["name"] = self.dir_name + "-" + str_index + self.file_type
        frame_entry["labels"] = self.create_label_list(xyxy, classes, confidence, ids)
        frame_entry["videoName"] = self.dir_name
        frame_entry["frameIndex"] = frame_index
        frame_entry["continuity"] = ids
        self.data.append(frame_entry)



    """
    Correctly formats
    """
    def create_label_list(self, xyxy:list, category:list, score:list, id:list):
        to_return = []
        for i in range(len(xyxy)):
            to_return.append(self.create_label_entry(xyxy[i], category[i], score[i], id[i]))
        return to_return
    
    def create_label_entry(self, xyxy:list, category:str, score:float, id:float):
        xydic = {
            "x1": xyxy[0],
            "x2": xyxy[2],
            "y1": xyxy[1],
            "y2": xyxy[3],
        }
        input_category = category
        if (category is float):
            input_category = self.cls_manager.get_subcategory(str(category))
        to_return = {
            "id": id,
            "category": input_category,
            "score": score,
            "box2d": xydic
        }
        print(category)
        return to_return

    def output_scalabel_detections(self, output_name="results"):

        # now write output to a file
        #json_data = json.load(self.data)
        pretty_print_json = pprint.pformat(self.data)
        pretty_print_json = pretty_print_json.replace("\'", '\"')

        with open(output_name+".json", 'w') as f:
            f.write(pretty_print_json)




            



        




