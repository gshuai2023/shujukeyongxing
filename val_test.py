from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolov8/runs/detect/train81/weights/best.pt') # 自己训练结束后的模型权重
    model.val(data='/root/autodl-tmp/yolov8/ultralytics/cfg/datasets/myyolo.yaml',
              split='val',
              imgsz=640,
              batch=16,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
