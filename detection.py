# File 
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from app.utils import detect_image, Load_Yolo_model
from app.configs import *

# image_path   = "./data/test/114_png.rf.6065afcc03288d34af746f3fd8fc4a1f.jpg"
image_path   = "./testing_photo/WIN_20221110_10_01_10_Pro.jpg"

yolo = Load_Yolo_model()
detect_image(yolo, image_path, "./output/a.jpg", input_size=YOLO_INPUT_SIZE, show=True, 
                CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))