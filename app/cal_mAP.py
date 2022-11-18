# file tạo hàm tính mAP
# ------------------------------------------------------------------------------
# Import thư viện cần thiết

import os
import tensorflow as tf
import numpy as np
from configs import *
from utils import *
import shutil
import json
import time

def voc_AP(precision, recall):

    recall.insert(0, 0.0)   # 0.0 ở đầu danh sách
    recall.append(1.0)      # 1.0 ở cuối danh sách      
    m_recall = recall[:]

    precision.insert(0, 0.0)    # 0.0 ở đầu danh sách
    precision.append(0.0)       # 0.0 ở cuối danh sách
    m_precision = precision[:]

    for index in range(len(m_precision) - 2, -1, -1):
        m_precision[index] = max(m_precision[index], m_precision[index + 1])

    index_list = []
    for index in range(1, len(m_recall)):
        if m_recall[index] != m_recall[index + 1]:
            index_list.append(index)

    ap = 0.0
    for index in index_list:
        ap += (m_recall[index] - m_recall[index - 1]) * m_precision[index]
    
    return ap, m_recall, m_precision

# Tạo hàm để tính mAP
def cal_mAP(Yolo, dataset, score_threshold=0.25, iou_threshold=0.5, TEST_INPUT_SIZE=TEST_INPUT_SIZE):
    
    MINOVERLAP = 0.5
    NUM_CLASS = read_class_names(TRAIN_CLASSES)

    # Path của Ground Truth <start>

    ground_truth_dir_path = 'mAP/ground_truth'
    if os.path.exists(ground_truth_dir_path):
        shutil.rmtree(ground_truth_dir_path)

    if not os.path.exists('mAP'):
        os.mkdir('mAP')
        os.mkdir(ground_truth_dir_path)

    # Path của Ground Truth <end>

    print(f'\nCalculating mAP{int(iou_threshold * 100)} . . .\n')

    class_ground_truth_counter = {} # dictionary chứa số lượng của class vật thể trong data
    for index in range(dataset.num_samples):
        annotation_dataset = dataset.annotations[index]

        # Lấy annotation trong data <start>

        original_image, bbox_data_ground_truth = dataset.parse_annotation(annotation_dataset, True)

        if len(bbox_data_ground_truth) == 0:
            bboxes_ground_truth = []
            classes_ground_truth = []
        else:
            bboxes_ground_truth = bbox_data_ground_truth[:, :4] # index đầu là danh sách vật thể, index sau là x, y, w, h?
            classes_ground_truth = bbox_data_ground_truth[:, 4] # index đầu là danh sách vật thể, index sau là class name?

        # ground_truth_path = os.path.join(ground_truth_dir_path, str(index) + '.txt')
        num_bbox_ground_truth = len(bboxes_ground_truth) # tổng số bounding box trong 1 sample

        # Lấy annotation trong data <end>



        bounding_boxes = [] # danh sách các bounding box

        for i in range(num_bbox_ground_truth):
            class_name = NUM_CLASS[classes_ground_truth[i]]
            xmin, ymin, xmax, ymax = list(map(str, bboxes_ground_truth[i]))
            bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})

            if class_name in class_ground_truth_counter:
                class_ground_truth_counter[class_name] += 1
            else:
                class_ground_truth_counter[class_name] = 1

            # bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'

        with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    ground_truth_classes = list(class_ground_truth_counter.keys())
    ground_truth_classes = sorted(ground_truth_classes)
    n_classes = len(ground_truth_classes)

    times = []
    json_predict = [[] for i in range(n_classes)]

    for index in range(dataset.num_samples):
        annotation_dataset = dataset.annotations[index]

        image_name = annotation_dataset[0].split('/')[-1]
        original_image, bbox_data_ground_truth = dataset.parse_annotation(annotation_dataset, True)

        image = image_preprocess(np.copy(original_image), [TEST_INPUT_SIZE, TEST_INPUT_SIZE])
        image_data = image[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            if tf.__version__ > '2.4.0':
                pred_bbox = Yolo(image_data)
            else:
                pred_bbox = Yolo.predict(image_data)
        else:
            raise ValueError("Unknown Framework!\n")
        t2 = time.time()

        times.append(t2 - t1)

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, TEST_INPUT_SIZE, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')


        for bbox in bboxes:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            class_name = NUM_CLASS[class_ind]
            score = '%.4f' % score
            xmin, ymin, xmax, ymax = list(map(str, coor))
            bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            json_predict[ground_truth_classes.index(class_name)].append({"confidence": str(score), "file_id": str(index), "bbox": str(bbox)})

    ms = sum(times) * 1000 / len(times)
    fps = 1000 / ms

    for class_name in ground_truth_classes:
        json_predict[ground_truth_classes.index(class_name)].sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
            json.dump(json_predict[ground_truth_classes.index(class_name)], outfile)

    # Calculate the AP for each class
    sum_AP = 0.0
    ap_dictionary = {}
    # open file to store the results
    with open("mAP/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(ground_truth_classes):
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
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / class_ground_truth_counter[class_name]
            #print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print(prec)

            ap, mrec, mprec = voc_AP(rec, prec)
            sum_AP += ap
            text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = [ '%.3f' % elem for elem in prec ]
            rounded_rec = [ '%.3f' % elem for elem in rec ]
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


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try: 
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            RuntimeError: print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')\n")
    return None



if __name__ == '__main__':
    main()
