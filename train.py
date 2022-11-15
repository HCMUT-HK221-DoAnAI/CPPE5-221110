# file huấn luyện mô hình
# ------------------------------------------------------------------------------
# Load các thư viện cần thiết
import os
import tensorflow as tf
from app.configs import *
from app.dataset import Dataset
from app.yolov3 import Create_Yolov3
from app.train_function import train_step, validate_step
from app.cal_mAP import cal_mAP

# Định nghĩa nội dung hàm main
def main():
    # Set up GPU    
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Khong co GPU."
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Tạo log để đọc bằng TensorBoard

    # Tạo dataset
    trainset = Dataset('train')
    testset = Dataset('test')

    # Tạo model, khai số step và epoch, khai loại optimizer
    yolo = Create_Yolov3()

    
    # ----------------------------------------------------------------
    # Vòng lặp epoch để train model
    for epoch in range(TRAIN_EPOCHS):
        # Gọi hàm dùng để train
        
        # Viết kết quả train vào trong log


        return


    # ----------------------------------------------------------------
    # Tạo model dùng để đo mAP
    # Load weight đã thu được sau khi train vào trong model
    # Gọi hàm tính mAP và xuất kết quả

    
# ------------------------------------------------------------------------------
# Gọi hàm main 
if __name__ == '__main__':
    main()