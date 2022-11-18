# file huấn luyện mô hình
# ------------------------------------------------------------------------------
# Load các thư viện cần thiết
import os
import tensorflow as tf
import shutil
from app.configs import *
from app.dataset import Dataset
from app.yolov3 import Create_Yolov3
from app.train_function import train_step, validate_step
from app.cal_mAP import cal_mAP

# Định nghĩa nội dung hàm main
def main():
    # Kiểm tra có GPU hay không
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "No GPU hardware available."
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Tạo log để đọc bằng TensorBoard
    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    # Tạo dataset
    trainset = Dataset('train')
    testset = Dataset('test')

    # Tạo model 
    yolo = Create_Yolov3()

    # Đọc vào weights nếu train từ yolov3 gốc hoặc train từ checkpoint
    if TRAIN_TRANSFER:
        Darknet = Create_Yolov3(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
        load_yolo_weights(Darknet, Darknet_weights) # use darknet weights

    if TRAIN_FROM_CHECKPOINT:
        try:
            yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
        except ValueError:
            print("Shapes are incompatible, transfering Darknet weights")
            TRAIN_FROM_CHECKPOINT = False

    if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo.layers[i].name)

    # Khai báo số step và epoch
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    # khai loại optimizer
    optimizer = tf.keras.optimizers.Adam()

    # val_loss khởi tạo
    best_val_loss = 1000
    
    
    # ----------------------------------------------------------------
    # Vòng lặp epoch để train model
    for epoch in range(TRAIN_EPOCHS):
        # Gọi hàm dùng để train
        for image_data, target in trainset:
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, \
                    conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], 
                  results[3], results[4], results[5]))
        
        # Lưu lại weight
        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue
        
        # Gọi hàm validate để tính validation loss
        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]

        # Viết kết quả validation vào torng log
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()

        # In ra terminal loss của epoch
        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, \
                total_val_loss:{:7.2f}\n\n"
                .format(giou_val/count, conf_val/count, prob_val/count, total_val/count))

        # Lưu lại weight của epoch
        if best_val_loss>total_val/count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = total_val/count


    # ----------------------------------------------------------------
    # Tạo model dùng để đo mAP
    mAP_model = Create_Yolov3(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    # Load weight đã thu được sau khi train vào trong model; gọi hàm tính mAP và xuất kết quả
    try:
        mAP_model.load_weights(save_directory) # use keras weights
        cal_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, 
                            iou_threshold=TEST_IOU_THRESHOLD)
    except UnboundLocalError:
        print("You don't have saved model weights to measure mAP, check TRAIN_SAVE_BEST_ONLY and \
                TRAIN_SAVE_CHECKPOINT lines in configs.py")
    
# ------------------------------------------------------------------------------
# Gọi hàm main 
if __name__ == '__main__':
    main()