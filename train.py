# file huấn luyện mô hình
# ------------------------------------------------------------------------------
# Load các thư viện cần thiết
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import shutil
from app.configs import *
from app.dataset import Dataset
from app.yolov3 import Create_Yolov3, compute_loss
from app.cal_mAP import cal_mAP
from app.utils import *

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
        Darknet = Create_Yolov3(input_size=YOLO_INPUT_SIZE, CLASSES = YOLO_COCO_CLASSES)
        load_yolo_weights(Darknet, YOLO_V3_WEIGHTS) # use darknet weights

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
    # Hàm để train và validate model
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0

            grid = 3
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            global_steps.assign_add(1)
            if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # Viết dữ liệu thống kê kết quả đối với từng step trong epoch
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            
        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    
    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=False)
            giou_loss=conf_loss=prob_loss=0

            grid = 3
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            
        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()
    # ----------------------------------------------------------------
    # Tạo model dùng để đo mAP
    mAP_model = Create_Yolov3(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    best_val_loss = 1000 # should be large at start
    # In kết quả cho từng step đối với từng epoch trong quá trình train (verbose=1)
    for epoch in range(TRAIN_EPOCHS):
        for image_data, target in trainset:
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))

        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue
        
        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
        # Ghi lại thống kê kết quả cho từng epoch
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()
            
        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val/count, conf_val/count, prob_val/count, total_val/count))
        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count))
            yolo.save_weights(save_directory)
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>total_val/count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = total_val/count
        if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
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