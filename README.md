# CPPE5

## Hướng dẫn train
+ Tải xuống [Data](https://universe.roboflow.com/dataset-format-conversion-zgvt9/datacppe5) dưới dạng Pascal VOC
+ Giải nén ta sẽ được thư mục data gồm 2 thư mục con là train và test. Sau đó thư mục data này sẽ được đặt vào trong folder chính
+ cd tới thư mục chính
+ Chạy trên terminal lệnh:
```python
    python3 create_model_data.py
```
+ Tiếp theo có thể vào app/config.py để tinh chỉnh một số tham số để huấn luyện như:

    Train transfer từ weight của YOLOv3:

        TRAIN_TRANSFER

    Train từ checkpoint được lưu từ lần train trước:

        TRAIN_FROM_CHECKPOINT

    Số lượng epoch:

        TRAIN_EPOCHS
    
    Chỉnh batch size:

        TRAIN_BATCH_SIZE

    Kích thước ảnh đầu vào:

        TRAIN_INPUT_SIZE

+ Sau đó cần gọi lệnh để bắt đầu quá trình train:

```python
    python3 train.py
```

## Hướng dẫn sử dụng detect image:
+ Comment lệnh detect_realtime
+ Uncomment lệnh detect_image
+ Sửa giá trị image_path thành đường dẫn tới ảnh cần detect.
+ Chạy trên Terminal:
```python
    python3 detection.py
```
## Hướng dẫn sử dụng detect realtime:
+ Comment lệnh detect_image
+ Uncomment lệnh detect_realtime
+ Chạy trên Terminal:
```python
    python3 detection.py
```
