# File 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from app.utils import Load_Yolo_model, detect_image, detect_realtime
from app.configs import *

image_path   = "C:/Users/binh/Documents/GitHub/CPPE5-221110/mobile-detection/input.jpeg"

yolo = Load_Yolo_model()
yolo.summary()
# detect_image(yolo, image_path, "./output/a.jpg", input_size=YOLO_INPUT_SIZE, show=True,
#              CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True,
                CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))