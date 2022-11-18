# File tạo các class liên quan đến tạo model yolov3
# ------------------------------------------------------------------------------
# Load các thư viện cần thiết
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    #INFO messages are not printed
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from keras.regularizers import L2
from app.configs import *

STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T/STRIDES).T
# Định nghĩa các nội dung liên quan
def convolutional(input_shape, kernel_size, filters, downsample=False, activation=True, bn=True):
    # Kiểm tra downsample
    if downsample is True:
        input_shape = ZeroPadding2D(((1,0),(0,1)))(input_shape)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1
    # Định nghĩa lớp Conv2D
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                  use_bias=not bn, kernel_regularizer=L2(l2=0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_shape)
    # Lớp conv có 2 loại, có và không BatchNormalization
    if bn:
        conv = BatchNormalization()(conv)
    if activation:
        conv = LeakyReLU(alpha=0.1)(conv)
    return conv

def residual_block(input_shape, filters_1, filters_2):
    short_cut = input_shape
    conv = convolutional(input_shape, 1, filters_1)
    conv = convolutional(conv, 3, filters_2)
    residual_output = short_cut + conv
    return residual_output

def upsample(input_shape):
    return tf.image.resize(input_shape, (input_shape.shape[1]*2,input_shape.shape[2]*2), method = 'nearest')

# Sắp xếp các lớp đã định nghĩa thành mạng darknet theo hình minh họa
def darknet53(input_data):
    # 2 lớp convolutional đầu tiên
    input_data = convolutional(input_data, 3, 32)
    input_data = convolutional(input_data, 3, 64, downsample = True)

    for i in range(1):
        input_data = residual_block(input_data, 32, 64)

    input_data = convolutional(input_data, 3, 128, downsample = True)

    for i in range(2):
        input_data = residual_block(input_data, 64, 128)

    input_data = convolutional(input_data, 3, 256, downsample = True)

    for i in range(8):
        input_data = residual_block(input_data, 128, 256)
    # Lưu lớp dự đoán tầng 1
    route_1 = input_data
    input_data = convolutional(input_data, 3, 512, downsample = True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 512)
    # Lưu lớp dự đoán tầng 2
    route_2 = input_data
    input_data = convolutional(input_data, 3, 1024, downsample = True)

    for i in range(4):
        input_data = residual_block(input_data, 512, 1024)
    # Trả về kết quả của lớp dự đoán tầng 1, tầng 2 và tầng cuối theo mô hình darknet53
    return route_1, route_2, input_data
# ----------------------------------------------------------------
# Dựa vào kết quả của darknet53 để rút ra kết quả ở 3 tỉ lệ ảnh
def YOLOv3(input_layer):
    # Lấy kết quả của 3 lớp dự đoán theo kiến trúc của darknet53
    route_1, route_2, conv = darknet53(input_layer)
    # Dẫn lớp conv (lớp cuối cùng) qua các lớp convolutional để có conv_lbbox
    conv = convolutional(conv, 1, 512)
    conv = convolutional(conv, 3, 1024)
    conv = convolutional(conv, 1, 512)
    conv = convolutional(conv, 3, 1024)
    conv = convolutional(conv, 1, 512)  # Checkpoint1 để xử lí tiếp cho lớp dự đoán tầng 2
    # Áp dụng 2 lớp Conv2D để được kết quả dự đoán cho tầng cuối.
    conv_lbbox = convolutional(conv, 3, 1024)
    # Dòng bên dưới sử dụng lớp convolutional với filter = 30 = B(3)*(4+1+C(5))
    # conv_lbbox dùng để dự đoán các vật thể lớn. Shape = [20,20,30]
    conv_lbbox = convolutional(conv_lbbox, 1, 30, activation=False, bn=False)

    # Từ lớp conv được chú thích là checkpoint1 ở trên ta áp dụng Conv2D và Upsampling2D để nối với route_2
    conv = convolutional(conv, 1, 256)
    conv = upsample(conv)
    # Sau đó ta nối với lớp dự đoán route_2
    conv = tf.concat([conv,route_2], axis=-1)
    # Tiếp tục dẫn lớp conv sau khi đã nối với route_2 qua các lớp convolutional để có conv_mbbox
    conv = convolutional(conv, 1, 256)
    conv = convolutional(conv, 3, 512)
    conv = convolutional(conv, 1, 256)
    conv = convolutional(conv, 3, 512)
    conv = convolutional(conv, 1, 256) # Checkpoint2 để xử lí tiếp cho lớp dự đoán tầng 1
    # Áp dụng 2 lớp Conv2D để được kết quả dự đoán cho tầng dự đoán thứ 2.
    conv_mbbox = convolutional(conv, 3, 512)
    # Dòng bên dưới sử dụng lớp convolutional với filter = 30 = B(3)*(4+1+C(5))
    # conv_mbbox dùng để dự đoán các vật thể trung bình. Shape = [40,40,30]
    conv_mbbox = convolutional(conv_mbbox, 1, 30, activation=False, bn=False)

    # Từ lớp conv được chú thích là checkpoint2 ở trên ta áp dụng Conv2D và Upsampling2D để nối với route_1
    conv = convolutional(conv, 1, 128)
    conv = upsample(conv)
    # Sau đó ta nối với lớp dự đoán route_1
    conv = tf.concat([conv,route_1], axis=-1)
    # Tiếp tục dẫn lớp conv sau khi đã nối với route_1 qua các lớp convolutional để có conv_sbbox
    conv = convolutional(conv, 1, 128)
    conv = convolutional(conv, 3, 256)
    conv = convolutional(conv, 1, 128)
    conv = convolutional(conv, 3, 256)
    conv = convolutional(conv, 1, 128)
    # Áp dụng 2 lớp Conv2D để được kết quả dự đoán cho tầng dự đoán thứ 1.
    conv_sbbox = convolutional(conv, 3, 256)
    # Dòng bên dưới sử dụng lớp convolutional với filter = 30 = B(3)*(4+1+C(5))
    # conv_sbbox dùng để dự đoán các vật thể trung nhỏ. Shape = [80,80,30]
    conv_sbbox = convolutional(conv_mbbox, 1, 30, activation=False, bn=False)    
    # Trả về kết quả của filters các lớp dự đoán
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def Create_Yolov3(input_size=640, channels=3, training=False):
    # Init dữ liệu đầu vào sử dụng hàm input của tf
    input_layer  = Input([input_size, input_size, channels])
    conv_tensors = YOLOv3(input_layer)
    # Init dữ liệu đầu ra sử dụng hàm YOLOv3
    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, i)
        if training: output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)
    # Tạo model bằng tf model
    YoloV3 = tf.keras.Model(input_layer, output_tensors)

    return YoloV3

def decode(conv_output, i=0):
    # where i = 0, 1 or 2 to correspond to the three grid scales  
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 10))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position     
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
    conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
    conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box 

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size,dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

    # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)