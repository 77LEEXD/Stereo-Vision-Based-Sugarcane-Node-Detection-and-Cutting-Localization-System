# Stereo Vision-Based Sugarcane Node Detection and Cutting Localization System

## 📌 Overview

This project implements a **real-time stereo vision system** for detecting sugarcane internodes and estimating their **3D geometric properties (depth, width, height)**.

It combines:

* **YOLOv8** for object detection
* **Stereo matching (SGBM + WLS)** for disparity estimation
* **3D reconstruction** using calibrated stereo parameters

The system is designed for **agricultural automation**, such as internode localization and intelligent cutting.

---

## 🚀 Features

* 🎯 Custom-trained YOLOv8 model for **internode detection**
* 📏 Real-time estimation of:

  * Depth
  * Width & Height (centimeters)
* 🧠 Robust disparity computation using **SGBM + WLS filtering**
* ⚡ Real-time performance (20–30 FPS with GPU)
* 📦 Fully reproducible (dataset + calibration included)

---

## 🧠 Method Pipeline

1. Stereo camera calibration (intrinsic & extrinsic)
2. Image rectification
3. YOLOv8 detection (left image)
4. Dense disparity estimation (SGBM)
5. WLS filtering for noise reduction
6. 3D point reconstruction via Q matrix
7. ROI-based point cloud filtering (5%–95% truncation)
8. Metric computation (depth, width, height)

---

## 📂 Project Structure

```id="r8p3t1"
yolo_project/
│
├── stereo_vision_node_localization.py   # Main system (real-time 3D measurement)
├── train4.py                            # YOLOv8 training script
│
├── dataset/
│   ├── data.yaml                        # Dataset configuration
│   ├── best.pt                          # Trained YOLOv8 weights
│   ├── stereo_calibration.npz          # Stereo calibration parameters
│   ├── train/                           # Training images & labels
│   ├── valid/                           # Validation set
│   └── test/                            # Test set
│
└── README.md
```

---

## ⚙️ Requirements

```bash id="c6n2f9"
pip install ultralytics opencv-python numpy torch
```

> ⚠️ Recommended:
>
> * GPU (CUDA)
> * OpenCV compiled with CUDA for better performance

---

## ▶️ Usage

### 1. Train the model

```bash id="n4x8q2"
python train4.py
```

### 2. Run real-time stereo measurement

```bash id="m7v2k1"
python stereo_vision_node_localization.py
```

---

## 📊 Output

For each detected internode:

* Size: `width × height` (cm)
* Bounding box visualization in real time

---

## ⚠️ Notes

* Measurement accuracy depends heavily on:

  * Camera calibration quality
  * Baseline accuracy 
* Ensure calibration parameters in `stereo_calibration.npz` are correct

---

## 📈 Future Work

* Replace SGBM with deep stereo models (e.g., RAFT-Stereo)
* Add temporal filtering for more stable measurements
* Perform quantitative error evaluation
* Integrate with robotic cutting systems

---

## 👤 Author

Xiaodong Li
