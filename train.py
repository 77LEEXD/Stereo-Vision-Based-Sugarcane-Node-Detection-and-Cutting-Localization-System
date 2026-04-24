import warnings
import os
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 确保保存目录存在（避免权限问题）
    save_dir = 'runs/detect/sugarcane_train'
    os.makedirs(save_dir, exist_ok=True)

    # 使用YOLOv8m预训练模型（可替换为yolov8l.pt提升精度）
    model = YOLO('yolov8m.pt')  # 或 'yolov8l.pt'

    # 训练模型
    results = model.train(
        # 数据配置
        data=r'E:\大学\人工智能\基于yolov8的甘蔗茎节检测\yolov8\datasets\sugarcane stem\data.yaml',

        # 训练轮次与学习率策略
        epochs=800,  # 增加训练轮次至800
        cos_lr=True,  # 启用余弦学习率衰减
        lr0=0.01,  # 初始学习率
        lrf=0.001,  # 最终学习率

        # 批量与图像尺寸
        batch=8,  # 保持批量大小8（避免显存不足）
        imgsz=480,  # 减小图像尺寸至480，提升小目标检测能力

        # 数据增强与正则化
        mosaic=1.0,  # 启用马赛克数据增强
        mixup=0.5,  # 启用mixup数据增强
        label_smoothing=0.1,  # 启用标签平滑，防止过拟合
        weight_decay=0.001,  # 增加权重衰减，提升模型泛化能力

        # 其他参数
        workers=8,
        device=0,
        optimizer='SGD',
        amp=True,
        cache=False,
        project=os.path.dirname(save_dir),
        name=os.path.basename(save_dir),
        save_period=50,  # 每50轮保存一次权重
        patience=50  # 连续50轮精度不提升则提前结束
    )