from typing import Any

import numpy as np
from keras_cv.backend import ops

class_ids = ["background", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
    "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
    "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ] 

class RetinaOutput:
    """
    Object which stores the RetinaNet object in an organized manner

    Parameters:
    pred:dict = output from the RetinaNet predictions

    Attributes:
    boxes:np.array = array of boxes
    probabilities:np.array = array of cls probabilities for each cls, sums up to approximately 1 (+/- 0.1)
    """

    def __init__(self, boxes, probabilities, current_size, old_size) -> None:

        self.boxes = []
        self.probabilities = []

        h_factor = old_size[0]/current_size[0]
        w_factor = old_size[1]/current_size[1]

        print(boxes.shape)

        i = 0

        for box in boxes:

            print(box)
            prob = probabilities[i]

            if (box[0] == 0 and box[1] ==0 and box[2] ==0 and box[3] == 0):
                break

            box[0] *= w_factor
            box[1] *= w_factor
            box[2] *= w_factor
            box[3] *= w_factor

            #print(f"{w_factor}+{h_factor} {old_b} vs {box}")

            self.boxes.append(box)
            self.probabilities.append(prob)

            i+= 1


        self.boxes = np.asarray(self.boxes)
        self.probabilities = np.asarray(self.probabilities)
        
        self.class_mapping = dict(zip(range(len(class_ids)), class_ids))


    def get_max_cls(self):
        """
        Converts cls probabilities into a discrete cls using argmax
        """

        return np.asarray(ops.argmax(self.probabilities, axis=-1))
    
    def get_cls_name(self):

        names = []

        cls = self.get_max_cls()

        if cls is None or cls.size < 1:
            return np.asarray([-1])

        try:
            for index in cls:
                names.append(self.class_mapping[index])
        except TypeError:
            names.append(cls)

        return np.asarray(names)



    def __str__(self) -> str:
        to_return = ""

        to_return += "boxes: "
        to_return += str(self.boxes)
        to_return += " probabilities: "
        to_return += str(self.probabilities)

        return to_return