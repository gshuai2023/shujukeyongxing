from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolov8/ultralytics/cfg/models/v8/yolov8s.yaml')
    model.load('/root/autodl-tmp/yolov8/yolov8s.pt') # loading pretrain weights
    model.train(data='/root/autodl-tmp/yolov8/ultralytics/cfg/datasets/myyolo.yaml', epochs=300, imgsz=640)







# # 这里如果需要预权重就写你的权重文件地址，没有预权重写cfg地址，写一个就够了
# model = YOLO("/root/autodl-tmp/yolov8/yolov8n.pt")
# # model = YOLO("/root/autodl-tmp/yolov8/ultralytics/cfg/models/v8/yolov8-qarepnext.yaml")




# model.train(model="/root/autodl-tmp/yolov8/ultralytics/cfg/models/v8/yolov8-qarepnext.yaml", data="/root/autodl-tmp/yolov8/ultralytics/cfg/datasets/myyolo.yaml", epochs=100, imgsz=640,device=0)