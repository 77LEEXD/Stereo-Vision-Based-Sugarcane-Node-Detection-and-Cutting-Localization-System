"""
Created on 2026/4/24

@author: LeeXD
"""

import cv2
import numpy as np
from ultralytics import YOLO

# =========================================================
# 1. 相机标定参数加载
# =========================================================
# 包含：
# - 内参（mtx）
# - 畸变系数（dist）
# - 外参（R, T）
calib_data = np.load(r"E:\大学\人工智能\基于yolov8的甘蔗茎节检测\数据\stereo_calibration.npz")

mtx_l, dist_l = calib_data["mtx_l"], calib_data["dist_l"]
mtx_r, dist_r = calib_data["mtx_r"], calib_data["dist_r"]
R, T = calib_data["R"], calib_data["T"]

# =========================================================
# 2. 图像分辨率设置（必须与标定一致）
# =========================================================
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080
image_size = (CAMERA_WIDTH, CAMERA_HEIGHT)

# =========================================================
# 3. 立体校正（Stereo Rectification）
# 作用：让左右图像“行对齐”，方便计算视差
# =========================================================
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l,
    mtx_r, dist_r,
    image_size,
    R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

# 生成去畸变 + 校正映射表（加速用）
map_lx, map_ly = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
)
map_rx, map_ry = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
)

# =========================================================
# 4. 加载YOLOv8模型（用于检测甘蔗茎节）
# =========================================================
model = YOLO(r"E:\大学\人工智能\基于yolov8的甘蔗茎节检测\数据\best.pt")

# YOLO推理尺寸（降低计算量）
YOLO_WIDTH, YOLO_HEIGHT = 640, 384

# =========================================================
# 5. SGBM参数初始化（立体匹配）
# 注意：只初始化一次，避免重复创建造成性能损耗
# =========================================================
window_size = 5
min_disp = 0
num_disp = 160  # 必须是16的倍数

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,

    # 匹配代价参数（控制平滑）
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,

    disp12MaxDiff=1,
    uniquenessRatio=10,

    # 去噪参数
    speckleWindowSize=100,
    speckleRange=32,

    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# =========================================================
# 6. 工具函数
# =========================================================
def rescale_box(box, from_size, to_size):
    """
    将YOLO检测框从缩放图像映射回原图尺寸
    """
    x_scale = to_size[0] / from_size[0]
    y_scale = to_size[1] / from_size[1]

    return [
        int(box[0] * x_scale),
        int(box[1] * y_scale),
        int(box[2] * x_scale),
        int(box[3] * y_scale)
    ]


def get_depth_from_bbox(disparity, Q, bbox):
    """
    根据检测框，从视差图中估计目标深度

    参数：
    - disparity: 全图视差
    - Q: 重投影矩阵
    - bbox: 检测框

    返回：
    - 深度（mm）或 None
    """
    x1, y1, x2, y2 = bbox

    # 防止越界
    h, w = disparity.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # 过滤过小区域（避免噪声）
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    # 提取ROI区域的视差
    roi_disp = disparity[y1:y2, x1:x2]

    # 过滤无效视差（负值/异常值）
    valid_disp = roi_disp[(roi_disp > 1) & (roi_disp < 150)]

    if len(valid_disp) < 50:
        return None

    # 使用中值代替均值（更抗噪）
    disp_median = np.median(valid_disp)

    # 取检测框中心点
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # 使用Q矩阵进行3D重建
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    Z = points_3D[cy, cx][2]  # Z即深度

    # 异常值过滤（单位：mm）
    if Z <= 0 or Z > 10000:
        return None

    return Z


# =========================================================
# 7. 主循环（实时处理）
# =========================================================
cap_l = cv2.VideoCapture(0)
cap_r = cv2.VideoCapture(1)

# 设置摄像头分辨率
cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

while True:
    ret_l, left_img = cap_l.read()
    ret_r, right_img = cap_r.read()

    if not ret_l or not ret_r:
        print("摄像头读取失败")
        break

    # =====================================================
    # 7.1 立体校正
    # =====================================================
    rectified_l = cv2.remap(left_img, map_lx, map_ly, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(right_img, map_rx, map_ry, cv2.INTER_LINEAR)

    # =====================================================
    # 7.2 计算全图视差（关键步骤）
    # =====================================================
    gray_l = cv2.cvtColor(rectified_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rectified_r, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

    # =====================================================
    # 7.3 YOLO检测（只用左图）
    # =====================================================
    yolo_img = cv2.resize(rectified_l, (YOLO_WIDTH, YOLO_HEIGHT))
    results = model(yolo_img, verbose=False)

    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():

            # 映射回原图坐标
            bbox = rescale_box(
                box,
                (YOLO_WIDTH, YOLO_HEIGHT),
                (CAMERA_WIDTH, CAMERA_HEIGHT)
            )

            # =================================================
            # 7.4 深度估计
            # =================================================
            depth = get_depth_from_bbox(disparity, Q, bbox)

            if depth is not None:
                print(f"检测到甘蔗茎节，距离：{depth:.2f} mm")

                # 绘制检测框
                cv2.rectangle(
                    rectified_l,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2
                )

                # 显示深度
                cv2.putText(
                    rectified_l,
                    f"{depth:.1f} mm",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    # =====================================================
    # 7.5 显示结果（缩小提高帧率）
    # =====================================================
    display_img = cv2.resize(rectified_l, (960, 540))
    cv2.imshow("Detection Result", display_img)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# 8. 资源释放
# =========================================================
cap_l.release()
cap_r.release()
cv2.destroyAllWindows()