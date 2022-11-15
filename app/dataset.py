# file tạo class Dataset
# ------------------------------------------------------------------------------
# Load các thư viện cần thiết
from app.configs import *

# Định nghĩa nội dung Class Dataset
def Dataset():
    # Hàm khởi tạo Dataset
    def __init__(self, dataset_type):
        # Thông tin dựa trên loại dataset
        if dataset_type == 'train':
            self.annot_path  = TRAIN_ANNOT_PATH
            self.input_sizes = TRAIN_INPUT_SIZE
            self.batch_size  = TRAIN_BATCH_SIZE
        elif dataset_type == 'test':
            self.annot_path  = TEST_ANNOT_PATH
            self.input_sizes = TEST_INPUT_SIZE
            self.batch_size  = TEST_BATCH_SIZE

        # Thông tin dựa trên annotations     
        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
    
    # Hàm đọc file để lấy data
    def load_annotations(self):
        final_annotations = []
        return final_annotations


    