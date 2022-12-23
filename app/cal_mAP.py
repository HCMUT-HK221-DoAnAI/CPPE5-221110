# file tạo hàm tính mAP
# ------------------------------------------------------------------------------
# Import thư viện cần thiết

import os
import numpy as np
import tensorflow as tf
# from tensorflow.python.saved_model import tag_constants
from app.dataset import Dataset
from app.yolov3 import Create_Yolov3
from app.utils import image_preprocess, postprocess_boxes, nms, read_class_names
from app.configs import *
import shutil
import json
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: 
        print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")


# Hàm dưới chưa rõ cách hoạt động
def voc_ap(recall, precision):

    recall.insert(0, 0.0)   # 0.0 ở đầu danh sách
    recall.append(1.0)      # 1.0 ở cuối danh sách
    m_recall = recall[:]

    precision.insert(0, 0.0)    # 0.0 ở đầu danh sách
    precision.append(0.0)       # 0.0 ở cuối danh sách
    m_precision = precision[:]

    for i in range(len(m_precision) - 2, -1, -1):
        m_precision[i] = max(m_precision[i], m_precision[i + 1])

    i_list = []
    for i in range(1, len(m_recall)):
        if m_recall[i] != m_recall[i - 1]:
            i_list.append(i) 

    ap = 0.0
    for i in i_list:
        ap += ((m_recall[i]-m_recall[i-1])*m_precision[i])
    return ap, m_recall, m_precision

# Tạo hàm để tính mAP
def cal_mAP(Yolo, dataset, score_threshold=0.25, iou_threshold=0.50, TEST_INPUT_SIZE=TEST_INPUT_SIZE):

    MINOVERLAP = 0.5 
    NUM_CLASS = read_class_names(TRAIN_CLASSES)

    # Path của Ground Truth <start>
    
    ground_truth_dir_path = 'mAP/ground-truth'
    if os.path.exists(ground_truth_dir_path):
        shutil.rmtree(ground_truth_dir_path)

    if not os.path.exists('mAP'):
        os.mkdir('mAP')

    os.mkdir(ground_truth_dir_path)

    # Path của Ground Truth <end>

    print(f'\nCalculating mAP{int(iou_threshold*100)}...\n')

    gt_counter_per_class = {}   # dictionary ghi nhận số lần xuất hiện của từng class vật thể trong data

    # Vòng lặp duyệt qua từng sample một trong tập ground_truth, tạo file .json ghi nhận các thông số về các bounding box có trong từng sample và đếm số lần xuất hiện của từng class vật thể trong toàn bộ dataset <start>

    for index in range(dataset.num_samples):
        annotation_dataset = dataset.annotations[index]

        # Lấy annotation trong data <start>

        original_image, bbox_data_gt = dataset.parse_annotation(annotation_dataset, True)

        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt = bbox_data_gt[:, :4] # index đầu là danh sách vật thể, index sau là x, y, w, h?
            classes_gt = bbox_data_gt[:, 4] # index đầu là danh sách vật thể, index sau là class name?

        ground_truth_path = os.path.join(ground_truth_dir_path, str(index) + '.txt')
        num_bbox_gt = len(bboxes_gt)    # tổng số bounding box trong 1 sample?

        # Lấy annotation trong data <end>

        bounding_boxes = [] # mảng các bounding box
        for i in range(num_bbox_gt):
            
            # Thêm bounding box trong nhãn của ảnh vào mảng bounding_boxes <start>

            class_name = NUM_CLASS[classes_gt[i]]
            xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
            bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})

            # Thêm bounding box trong nhãn của ảnh vào mảng bounding_boxes <end>

            # Đếm số lượng xuất hiện của từng class vật thể trong sample hiện tại <start>

            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1
            bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'

        # Đếm số lượng xuất hiện của từng class vật thể trong sample hiện tại <start>
        
        with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
            json.dump(bounding_boxes, outfile)

        # Đếm số lượng xuất hiện của từng class vật thể trong sample hiện tại <end>

    # Vòng lặp duyệt qua từng sample một trong tập ground_truth, tạo file .json ghi nhận các thông số về các bounding box có trong từng sample và đếm số lần xuất hiện của từng class vật thể trong toàn bộ dataset <end>

    gt_classes = list(gt_counter_per_class.keys())  # Danh sách các class có trong data
    gt_classes = sorted(gt_classes)                 # Sắp xếp tên class theo alphabet
    n_classes = len(gt_classes)                     # Số lượng class có trong data

    times = []
    json_pred = [[] for i in range(n_classes)]

    # Vòng lặp quét qua từng sample 1 và ghi nhận dự đoán về bounding box của model trên từng sample <start>

    for index in range(dataset.num_samples):
        annotation_dataset = dataset.annotations[index]

        # image_name = annotation_dataset[0].split('/')[-1]
        original_image, bbox_data_gt = dataset.parse_annotation(annotation_dataset, True)
        
        image = image_preprocess(np.copy(original_image), [TEST_INPUT_SIZE, TEST_INPUT_SIZE])   # Scale ảnh về kích thước chung của model
        image_data = image[np.newaxis, ...].astype(np.float32)

        t1 = time.time()

        pred_bbox = Yolo(image_data)
        
        t2 = time.time()
        
        times.append(t2-t1)     # Ghi nhận thời gian dự đoán từng sample của model
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, TEST_INPUT_SIZE, score_threshold) # Xử lý các bounding boxes đã được dự đoán
        bboxes = nms(bboxes, iou_threshold, method='nms')   # Dùng non-max suppression để tạo nên output cuối của model

        # Vòng lặp ghi nhận dự đoán của model vào mảng json_pred <start>

        for bbox in bboxes:
            coor = np.array(bbox[:4], dtype=np.int32)   # Tọa độ box
            score = bbox[4]                             # Độ tin cậy của box
            class_ind = int(bbox[5])
            class_name = NUM_CLASS[class_ind]
            score = '%.4f' % score
            xmin, ymin, xmax, ymax = list(map(str, coor))
            bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            json_pred[gt_classes.index(class_name)].append({"confidence": str(score), "file_id": str(index), "bbox": str(bbox)})

        # Vòng lặp ghi nhận dự đoán của model vào mảng json_pred <end>

    # Vòng lặp quét qua từng sample 1 và ghi nhận dự đoán về bounding box của model trên từng sample <end>

    ms = sum(times)/len(times)*1000
    fps = 1000 / ms

    # Xuất kết quả dự đoán bounding box của model vào file .json theo độ tin cậy giảm dần của các box <start>

    for class_name in gt_classes:
        json_pred[gt_classes.index(class_name)].sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
            json.dump(json_pred[gt_classes.index(class_name)], outfile)

    # Xuất kết quả dự đoán bounding box của model vào file .json theo độ tin cậy giảm dần của các box <end>

    # Calculate the AP for each class
    sum_AP = 0.0
    ap_dictionary = {}
    # open file to store the results
    with open("mAP/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            # Load predictions of that class
            predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
            predictions_data = json.load(open(predictions_file))

            # Assign predictions to ground truth objects
            nd = len(predictions_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                # assign prediction to ground truth object if any
                #   open ground-truth with that file_id
                gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load prediction bounding-box
                bb = [ float(x) for x in prediction["bbox"].split() ] # bounding box of prediction
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [ float(x) for x in obj["bbox"].split() ] # bounding box of ground truth
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign prediction as true positive/don't care/false positive
                if ovmax >= MINOVERLAP:# if ovmax > minimum overlap
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            #print(tp)
            recall = tp[:]
            for idx, val in enumerate(tp):
                recall[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #print(recall)
            precision = tp[:]
            for idx, val in enumerate(tp):
                precision[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print(precision)

            ap, m_recall, mprec = voc_ap(recall, precision)
            sum_AP += ap
            text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = [ '%.3f' % elem for elem in precision ]
            rounded_rec = [ '%.3f' % elem for elem in recall ]
            # Write to results.txt
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP*100, fps)
        results_file.write(text + "\n")
        print(text)
        
        return mAP*100

if __name__ == '__main__':       
    yolo = Create_Yolov3(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use custom weights
        
    testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
    cal_mAP(yolo, testset, score_threshold=0.05, iou_threshold=0.50, TEST_INPUT_SIZE=YOLO_INPUT_SIZE)