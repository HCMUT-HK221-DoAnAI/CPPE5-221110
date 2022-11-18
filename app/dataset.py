# file tạo class Dataset
# ------------------------------------------------------------------------------
# Load các thư viện cần thiết
import os
import cv2
import numpy as np
from app.utils import read_class_names, image_preprocess
from app.configs import *

# Định nghĩa nội dung Class Dataset
class Dataset(object):
    def __init__(self, dataset_type):
        self.annot_path  = TRAIN_ANNOT_PATH if dataset_type == 'train' else TEST_ANNOT_PATH
        self.input_sizes = TRAIN_INPUT_SIZE if dataset_type == 'train' else TEST_INPUT_SIZE
        self.batch_size  = TRAIN_BATCH_SIZE if dataset_type == 'train' else TEST_BATCH_SIZE
        
        self.train_input_sizes = TRAIN_INPUT_SIZE
        self.strides = np.array(YOLO_STRIDES)
        self.classes = read_class_names(TRAIN_CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = (np.array(YOLO_ANCHORS).T/self.strides).T
        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
    
    def load_annotations(self):
        final_annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.read().splitlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)

        for annotation in annotations:
            line = annotation.split()
            image_path, index = "", 1
            for i, one_line in enumerate(line):
                if not one_line.replace(",","").isnumeric():
                    if image_path != "": image_path += " "
                    image_path += one_line
                else:
                    index = i
                    break
            if not os.path.exists(image_path):
                raise KeyError("%s does not exist ... " %image_path)
            if TRAIN_LOAD_IMAGES_TO_RAM:
                image = cv2.imread(image_path)
            else:
                image = ''
            final_annotations.append([image_path, line[index:], image])
        return final_annotations

    def __iter__(self):
        return self

    def parse_annotation(self, annotation, mAP = 'False'):
        if TRAIN_LOAD_IMAGES_TO_RAM:
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path)

        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])

        if mAP == True:
            return image, bboxes

        image, bboxes = image_preprocess(np.copy(image), [self.input_sizes, self.input_sizes], np.copy(bboxes))
        return image, bboxes

    def __len__(self):
        return self.num_batchs