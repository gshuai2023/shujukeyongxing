from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolov8/runs/detect/train40/weights/best.pt') # 自己训练结束后的模型权重
    model.val(data='/root/autodl-tmp/yolov8/ultralytics/cfg/datasets/myyolo.yaml',
              split='test',
              imgsz=640,
              batch=16,
              project='runs/test',
              name='exp',
              )
   