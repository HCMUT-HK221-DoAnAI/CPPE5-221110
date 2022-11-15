# file tạo các class liên quan đến tạo model yolov3
# ------------------------------------------------------------------------------
# Load các thư viện cần thiết
import tensorflow as tf
from keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D

# Định nghĩa các nội dung liên quan
def convolutional(input_layer): 
    conv = Conv2D()
    # Lớp conv có 2 loại, có và không BatchNormalization
    return conv

def residual_block(input_layer):
    short_cut = input_layer
    conv = convolutional(input_layer)
    conv = convolutional(conv)
    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    return

# Sắp xếp các lớp đã định nghĩa thành mạng darknet theo hình minh họa
def darknet53(input_data):
    input_data = convolutional()
    input_data = convolutional()

    for i in range(1):
        input_data = residual_block()

    input_data = convolutional()

    for i in range(2):
        input_data = residual_block()

    input_data = convolutional()

    for i in range(8):
        input_data = residual_block()

    route_1 = input_data
    input_data = convolutional()

    for i in range(8):
        input_data = residual_block()

    route_2 = input_data
    input_data = convolutional()

    for i in range(4):
        input_data = residual_block()

    return route_1, route_2, input_data
# ----------------------------------------------------------------
# Dựa vào kết quả của darknet53 để rút ra kết quả ở 3 tỉ lệ ảnh
def YOLOv3(input_layer):
    route_1, route_2, conv = darknet53(input_layer)

    # Dẫn lớp conv (lớp cuối cùng) qua các lớp convolutional để có conv_lbbox
    conv_lbbox = None
    # Dẫn route_2 qua các lớp convolutional để có conv_mbbox
    conv_mbbox = None
    # Dẫn route_1 qua các lớp convolutional để có conv_sbbox
    conv_sbbox = None
    
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def Create_Yolov3():
    # Init dữ liệu đầu vào sử dụng hàm input của tf
    input_layer = input()
    # Init dữ liệu đầu ra sử dụng hàm YOLOv3
    output_tensors = None
    # Tạo model bằng tf model
    YoloV3 = tf.keras.Model(input_layer, output_tensors)

    return YoloV3