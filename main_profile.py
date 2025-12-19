import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import train

if __name__ == '__main__':
    # 是否支持6通道输入 一定与模型匹配
    train.IR_support = True

    # choose your yaml file
    model = YOLO('runs/train/VEDAI-RIFusion-ADD-10244/weights/best.pt')
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    model.fuse()