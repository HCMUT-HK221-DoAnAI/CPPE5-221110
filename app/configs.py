# YOLOv3 options
YOLO_INPUT_SIZE = 640

# Train options
TRAIN_LOGDIR = "log"
TRAIN_ANNOT_PATH = "./model_data/train.txt"
TRAIN_CLASSES = "./model_data/names.txt"
TRAIN_BATCH_SIZE = 2
TRAIN_INPUT_SIZE = 640
TRAIN_EPOCHS = 10
YOLO_STRIDES = [8, 16, 32]
YOLO_ANCHORS = [[[10,  13], [16,   30], [33,   23]],
                [[30,  61], [62,   45], [59,  119]],
                [[116, 90], [156, 198], [373, 326]]]
# TEST options
TEST_INPUT_SIZE = 640
TEST_ANNOT_PATH = "./model_data/test.txt"
TEST_BATCH_SIZE = 4