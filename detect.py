import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import train

# BILIBILI UP 魔傀面具
# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    #关闭红外通道   下面的sourceIR不注释掉也不会报错
    train.IR_support = True
#   runs/train/v8n-5.21-ADD/weights/best.pt
#   runs/train/v8n-5.22-RIFusion-CNN-ADDCNN/weights/best.pt
    model = YOLO('runs/train/v8n-5.22-RIFusion-CNN-ADDCNN/weights/best.pt') # select your model.pt path
    model.predict(source="dataset/OBBCrop/images/test/00001.jpg",
                  sourceIR="dataset/OBBCrop/imageIR/test/00001.jpg",
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  conf=0.45,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )