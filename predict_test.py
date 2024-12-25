from ultralytics import YOLO

# 加载训练好的模型，改为自己的路径
model = YOLO("/root/autodl-tmp/yolov8/runs/detect/train83/weights/best.pt")
# 修改为自己的图像或者文件夹的路径
source = "/root/autodl-tmp/yolov8/yuce/MVI_40772__img00757.jpg"
# 运行推理，并附加参数
model.predict(source, save=True)





# model = YOLO("/root/autodl-tmp/yolov8/runs/detect/train27/weights/best.pt")  # 权重

# results = model("/root/autodl-tmp/yolov8/datasets/UA-DETRAC2/images/train/MVI_40131__img00037.jpg")  # 预测的图片或文件夹