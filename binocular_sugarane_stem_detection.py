import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------- 1. 初始化参数 ----------------------
# 加载标定参数
calib_data = np.load(r"E:\大学\人工智能\基于yolov8的甘蔗茎节检测\数据\stereo_calibration.npz")
mtx_l, dist_l = calib_data["mtx_l"], calib_data["dist_l"]
mtx_r, dist_r = calib_data["mtx_r"], calib_data["dist_r"]
R, T = calib_data["R"], calib_data["T"]

# 修改点1：使用摄像头原生分辨率1920x1080
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080
image_size = (CAMERA_WIDTH, CAMERA_HEIGHT)  # 更新标定图像尺寸

# 计算立体校正参数
R1, R2, P1, P2, Q, validROI_l, validROI_r = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

# 生成重映射表
map_lx, map_ly = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1)
map_rx, map_ry = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1)

# 加载YOLO模型
model = YOLO(r"E:\大学\人工智能\基于yolov8的甘蔗茎节检测\数据\best.pt")

# 双目测距参数
baseline = np.linalg.norm(T)  # 基线距离(mm)
focal_length = (mtx_l[0, 0] + mtx_r[0, 0]) / 2  # 平均焦距

# 修改点2：YOLO输入尺寸（保持16:9比例）
YOLO_WIDTH, YOLO_HEIGHT = 640, 384


# ---------------------- 2. 工具函数 ----------------------
def rescale_box(box, from_size, to_size):
    """将检测框坐标从一个尺寸转换到另一个尺寸"""
    x_scale = to_size[0] / from_size[0]
    y_scale = to_size[1] / from_size[1]
    return [
        int(box[0] * x_scale),
        int(box[1] * y_scale),
        int(box[2] * x_scale),
        int(box[3] * y_scale)
    ]


def calculate_distance(rectified_l, rectified_r, bbox, Q):
    """计算目标距离（修改点3：优化视差计算）"""
    x1, y1, x2, y2 = bbox

    # 边界检查
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(rectified_l.shape[1] - 1, x2), min(rectified_l.shape[0] - 1, y2)

    # 确保检测区域有效
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    # 裁剪目标区域（左右图使用相同区域）
    crop_l = rectified_l[y1:y2, x1:x2]
    crop_r = rectified_r[y1:y2, x1:x2]

    # 转换为灰度图
    gray_l = cv2.cvtColor(crop_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)

    # 修改点4：改进的SGBM参数
    window_size = 5
    min_disp = 0
    num_disp = 160 - min_disp
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 计算视差
    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

    # 计算有效视差区域
    mask = (disparity > min_disp) & (disparity < num_disp)
    if not np.any(mask):
        return None

    mean_disp = np.mean(disparity[mask])
    distance = (baseline * focal_length) / (mean_disp + 1e-6)  # 避免除零

    return distance


# ---------------------- 3. 主循环 ----------------------
cap_l = cv2.VideoCapture(0)
cap_r = cv2.VideoCapture(1)

# 修改点5：设置摄像头分辨率
cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

while True:
    # 读取原始图像
    ret_l, left_img = cap_l.read()
    ret_r, right_img = cap_r.read()
    if not ret_l or not ret_r:
        break

    # 立体校正
    rectified_l = cv2.remap(left_img, map_lx, map_ly, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(right_img, map_rx, map_ry, cv2.INTER_LINEAR)

    # 修改点6：调整尺寸供YOLO检测
    yolo_img_l = cv2.resize(rectified_l, (YOLO_WIDTH, YOLO_HEIGHT))
    yolo_img_r = cv2.resize(rectified_r, (YOLO_WIDTH, YOLO_HEIGHT))

    # YOLO检测
    results_l = model(yolo_img_l)
    results_r = model(yolo_img_r)

    # 处理检测结果
    for result_l, result_r in zip(results_l, results_r):
        for box_l, box_r in zip(result_l.boxes.xyxy.cpu().numpy(), result_r.boxes.xyxy.cpu().numpy()):
            # 将检测框坐标转换回原图尺寸
            bbox_l = rescale_box(box_l, (YOLO_WIDTH, YOLO_HEIGHT), (CAMERA_WIDTH, CAMERA_HEIGHT))
            bbox_r = rescale_box(box_r, (YOLO_WIDTH, YOLO_HEIGHT), (CAMERA_WIDTH, CAMERA_HEIGHT))

            # 计算距离（使用左图坐标）
            distance = calculate_distance(rectified_l, rectified_r, bbox_l, Q)

            if distance is not None:
                print(f"检测到甘蔗茎节，距离：{distance:.2f}mm")

                # 在原始图像上绘制结果
                cv2.rectangle(rectified_l, (bbox_l[0], bbox_l[1]), (bbox_l[2], bbox_l[3]), (0, 255, 0), 2)
                cv2.putText(rectified_l, f"{distance:.2f}mm",
                            (bbox_l[0], bbox_l[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 修改点7：显示缩放后的结果（提高显示效率）
    display_img = cv2.resize(rectified_l, (960, 540))  # 缩小50%显示
    cv2.imshow("Detection Result", display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_l.release()
cap_r.release()
cv2.destroyAllWindows()