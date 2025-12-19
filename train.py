import warnings, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV8V10配置文件.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch

# ultralytics/cfg/models/yaml-supportIR/yolov8_MPF.yaml
# ultralytics/cfg/models/v8/yolov8.yaml

#是否支持额外的红外光图像，除非需要做红外与可见光中期融合否则绝对不要开启
# https://blog.csdn.net/qq_32575047/article/details/144946303  改动参考
# ultralytics/data/base.py  ->  load_image中进行了改动，此外请全局搜索IR_support
# 具体方法为 正常加载RGB图像，然后使用'imageIR'替换掉RGB图像路径中的‘images’，从而使IR图像也加载，之后将IR图像dstack在RGB图像之后
# 模型处理数据方式在ultralytics/nn/tasks.py  ->  _predict_once    若更改模型结构YAML导致报错多半错误在这里
IR_support = True

if __name__ == '__main__':
    model = YOLO(model='ultralytics\cfg\models\yaml-supportIR\yolov8-obb-Fusion.yaml',
                 task=None,             # 会根据yaml文件自适应  segment  classify  pose  detect
                 verbose=False          # 用于控制模型运行时的日志输出级别
                 )
    # model.load('yolov8n.pt') # loading pretrain weights
    # dataset/OBBCrop/drone2.yaml
    # dataset/data.yaml
    model.train(data='dataset/VEDAI/data.yaml',
                cache=False,
                imgsz=1024,
                epochs=300,
                batch=8,
                close_mosaic=1,
                workers=4,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                optimizer='AdamW',  # using SGD
                # device='0,1', # 指定显卡和多卡训练参考<YOLOV8V10配置文件.md>下方常见错误和解决方案
                patience=50,  # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='VEDAI-RIFusion-ADD-1024',
                )
