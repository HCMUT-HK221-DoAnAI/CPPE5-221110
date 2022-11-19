# File tạo các class liên quan đến tạo model yolov3
# ------------------------------------------------------------------------------
# Load các thư viện cần thiết
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    #INFO messages are not printed
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization
from keras.regularizers import L2
from app.configs import *

def read_class_names(class_filename):
    names = {}
    with open(class_filename) as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

class BatchNormalization(BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T/STRIDES).T
# Định nghĩa các nội dung liên quan
def convolutional(input_shape, filters_shape, downsample=False, activation=True, bn=True):
    # Kiểm tra downsample
    if downsample is True:
        input_shape = ZeroPadding2D(((1,0),(0,1)))(input_shape)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1
    # Định nghĩa lớp Conv2D
    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=L2(l2=0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_shape)
    # Lớp conv có 2 loại, có và không BatchNormalization
    if bn:
        conv = BatchNormalization()(conv)
    if activation:
        conv = LeakyReLU(alpha=0.1)(conv)
    return conv

def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))
    residual_output = short_cut + conv
    return residual_output

def upsample(input_shape):
    return tf.image.resize(input_shape, (input_shape.shape[1]*2,input_shape.shape[2]*2), method = 'nearest')

# Sắp xếp các lớp đã định nghĩa thành mạng darknet theo hình minh họa
def darknet53(input_data):
    # 2 lớp convolutional đầu tiên
    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)

    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)
    # Lưu lớp dự đoán tầng 1
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)
    # Lưu lớp dự đoán tầng 2
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)
    # Trả về kết quả của lớp dự đoán tầng 1, tầng 2 và tầng cuối theo mô hình darknet53
    return route_1, route_2, input_data
# ----------------------------------------------------------------
# Dựa vào kết quả của darknet53 để rút ra kết quả ở 3 tỉ lệ ảnh
def YOLOv3(input_layer, NUM_CLASS):
    # Lấy kết quả của 3 lớp dự đoán theo kiến trúc của darknet53
    route_1, route_2, conv = darknet53(input_layer)
    # Dẫn lớp conv (lớp cuối cùng) qua các lớp convolutional để có conv_lbbox
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))  # Checkpoint1 để xử lí tiếp cho lớp dự đoán tầng 2
    # Áp dụng 2 lớp Conv2D để được kết quả dự đoán cho tầng cuối.
    conv_lbbox = convolutional(conv, (3, 3, 512, 1024))
    # Dòng bên dưới sử dụng lớp convolutional với filter = 30 = B(3)*(4+1+C(5))
    # conv_lbbox dùng để dự đoán các vật thể lớn. Shape = [20,20,30]
    conv_lbbox = convolutional(conv_lbbox, (1, 1, 1024, 3*(NUM_CLASS + 5)), activation=False, bn=False)

    # Từ lớp conv được chú thích là checkpoint1 ở trên ta áp dụng Conv2D và Upsampling2D để nối với route_2
    conv = convolutional(conv, (1, 1,  512,  256))
    conv = upsample(conv)
    # Sau đó ta nối với lớp dự đoán route_2
    conv = tf.concat([conv,route_2], axis=-1)
    # Tiếp tục dẫn lớp conv sau khi đã nối với route_2 qua các lớp convolutional để có conv_mbbox
    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256)) # Checkpoint2 để xử lí tiếp cho lớp dự đoán tầng 1
    # Áp dụng 2 lớp Conv2D để được kết quả dự đoán cho tầng dự đoán thứ 2.
    conv_mbbox = convolutional(conv, (3, 3, 256, 512))
    # Dòng bên dưới sử dụng lớp convolutional với filter = 30 = B(3)*(4+1+C(5))
    # conv_mbbox dùng để dự đoán các vật thể trung bình. Shape = [40,40,30]
    conv_mbbox = convolutional(conv_mbbox, (1, 1, 512, 3*(NUM_CLASS + 5)), activation=False, bn=False)

    # Từ lớp conv được chú thích là checkpoint2 ở trên ta áp dụng Conv2D và Upsampling2D để nối với route_1
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    # Sau đó ta nối với lớp dự đoán route_1
    conv = tf.concat([conv,route_1], axis=-1)
    # Tiếp tục dẫn lớp conv sau khi đã nối với route_1 qua các lớp convolutional để có conv_sbbox
    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    # Áp dụng 2 lớp Conv2D để được kết quả dự đoán cho tầng dự đoán thứ 1.
    conv_sbbox = convolutional(conv, (3, 3, 128, 256))
    # Dòng bên dưới sử dụng lớp convolutional với filter = 30 = B(3)*(4+1+C(5))
    # conv_sbbox dùng để dự đoán các vật thể trung nhỏ. Shape = [80,80,30]
    conv_sbbox = convolutional(conv_sbbox, (1, 1, 256, 3*(NUM_CLASS +5)), activation=False, bn=False)
    # Trả về kết quả của filters các lớp dự đoán
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def Create_Yolov3(input_size=640, channels=3, training=False, CLASSES=YOLO_COCO_CLASSES):
    NUM_CLASS = len(read_class_names(CLASSES))
    # Init dữ liệu đầu vào sử dụng hàm input của tf
    input_layer  = Input([input_size, input_size, channels])
    conv_tensors = YOLOv3(input_layer, NUM_CLASS)
    # Init dữ liệu đầu ra sử dụng hàm YOLOv3
    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training: output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)
    # Tạo model bằng tf model
    YoloV3 = tf.keras.Model(input_layer, output_tensors)

    return YoloV3

def decode(conv_output, NUM_CLASS, i=0):
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # Độ dời của điểm trọng tâm của bbox
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Độ dời của chiều dài và chiều rộng bbox
    conv_raw_conf = conv_output[:, :, :, :, 4:5] # Độ tin cậy của bbox dự đoán
    conv_raw_prob = conv_output[:, :, :, :, 5: ] # Xác suất chính xác của bbox
    
    # Vẽ lưới với các kích thước đầu ra là 20,40,80
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size,dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Tính vị trí trọng tâm của bounding box
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Tính chiều dài và chiều rộng của bounding box
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    # Tính độ tin cậy của bbox dự đoán
    pred_conf = tf.sigmoid(conv_raw_conf) 
    # Tính xác xuất chính xác của bbox
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Tính giá trị của IOU giữa 2 bounding box
    iou = inter_area / union_area

    # Tính toạ độ của góc trái trên và góc phải dưới
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Tính diện tích
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Tính giá trị của GIOU
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=YOLO_COCO_CLASSES):
    NUM_CLASS = len(read_class_names(CLASSES))
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # Tính giá trị của IOU đối với box thật
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < YOLO_IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Tính giá trị của hàm loss của sự tin cậy
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss