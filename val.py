import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import train

# BILIBILI UP 魔傀面具
# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

# 精度小数点保留位数修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第五点
# 最终论文的参数量和计算量统一以这个脚本运行出来的为准

# 必须与模型的模态设定一致
train.IR_support = True

if __name__ == '__main__':
    model = YOLO('runs/train/VEDAI-1024/weights/best.pt') # 选择训练好的权重路径
    model.val(data='dataset/VEDAI/data.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=1024,
              batch=8,
              # iou=0.53,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='Vedai no change',
              )

#0.792 -- 0.784
#0.794 -- 0.786
#0.798 -- 0.788
#0.8   -- 0.791
#0.805 -- 0.794
#0.810 -- 0.798
#0.821 -- 0.805
#0.825 -- 0.807
#0.831 -- 0.809
#0.837 -- 0.811 0.609
#0.839 -- 0.812 0.611              0.840  --  0.809
#0.84114  0.813 0.613
#0.84291  0.814
#0.84545  0.814 0.615
#0.84575  0.814 0.615
#0.84599  0.814