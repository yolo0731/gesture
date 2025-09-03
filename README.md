# 手势与手写识别（Python）

本项目基于 OpenCV、MediaPipe、PyTorch 与 TensorFlow，实现：

- 手写数字识别（摄像头空中书写）
- 手写字母识别（摄像头空中书写）
- 数字手势识别（ASL 数字）
- 字母手势识别（ASL 字母，含 J/Z 轨迹）
- 动态手势识别（Click/Stop/Rotate/No）

项目包含一个 PyQt5 交互界面，方便在同一窗口中体验以上能力。

## 运行环境

- Python 3.9（推荐）
- 一键创建 Conda 环境（CPU 默认）：

```
conda env create -f environment.yml
conda activate gesture
```

- 如果有 NVIDIA GPU 并希望用 CUDA 加速：

```
conda env create -f environment.gpu.yml
conda activate gesture
```

- 也可以使用 pip 安装（可选）：`pip install -r requirements.txt`

注意：动态/手写识别需要加载本地模型文件（较大，默认被 .gitignore 忽略，不随仓库提交）。见“模型文件”一节。

## 快速上手

1) 启动交互界面：

方式A（推荐，免配置）：
```
python run.py
```

方式B（使用包运行）：
```
python -m gesture_app
```

- 窗口内按钮功能：
  - 1. 手写数字识别：空中书写数字，实时识别与显示
  - 2. 手写字母识别：空中书写字母，实时识别与显示
  - 3. 数字手势识别：对准摄像头做 ASL 数字手势
  - 4. 字母手势识别：对准摄像头做 ASL 字母手势
  - 5. 动态手势识别：Click/Stop/Rotate/No
  - 6. 动态字母手势：J/Z（轨迹识别）

2) 摄像头索引适配（可选）：

不同设备的摄像头索引可能不同（常见 0/1）。仓库提供辅助脚本自动探测并批量替换：

```
python scripts/detect_and_update_camera_index.py
```

- 仅查看将要修改的内容：`--dry-run`
- 指定扫描上限：`--max 20`
- 跳过探测，强制使用某索引：`--force 1`

## 模型文件

以下模型目录/文件较大，已被 `.gitignore` 忽略，不会上传到 GitHub。请在本地放置到 `models/` 目录：

- 手写字母（Keras）：`models/action.h5_3/`（SavedModel 目录）
- 动态手势（Keras）：`models/CSRN_Model_2/`（SavedModel 目录）
- 手写数字（PyTorch）：`models/MNIST1.pth`
- 手写字母（PyTorch）：`models/EMNIST2_5.18.pth`

若缺少以上文件，对应功能会无法加载模型或识别失败。

## 目录说明（清理后）

- 核心交互与 UI（包内）：
  - `gesture_app/jiemianshixian.py`（应用窗口）
  - `gesture_app/interaction_6.py`、`gesture_app/ui/interaction_6.ui`
  - 模块启动：`python -m gesture_app`
- 手写/手势模块（包内）：
  - `gesture_app/written_number.py`、`gesture_app/written_letter.py`
  - `gesture_app/number_gesture.py`、`gesture_app/letter_gesture.py`
  - `gesture_app/HandTrackingModule.py`、`gesture_app/HandTrackingModule_letter.py`
  - `gesture_app/letter_interpre.py`、`gesture_app/Finger.py`、`gesture_app/Landmark.py`、`gesture_app/definition.py`
- 深度学习（包内）：
  - `gesture_app/cnntrain_mnist.py`（数字模型结构/训练）
  - `gesture_app/duan_validation_2.py`（字母模型结构/训练）
- 动态手势：
  - `dg_prediction_CSRN.py`
- 脚本与工具：
  - `scripts/detect_and_update_camera_index.py`
  - `scripts/test_camera.py`（摄像头可用性测试）

数据集与模型目录默认忽略：`data/`、`models/`（含 SavedModel/权重文件）等。

附：关键点可视化 Demo：
```
python -m gesture_app.demos.hand_keypoints
```

## 常见问题

- 摄像头打不开或黑屏：请运行 `scripts/detect_and_update_camera_index.py` 自动探测索引，或检查是否被其他应用占用。
- Windows 摄像头后端：部分代码在 Linux 使用 `cv2.CAP_V4L2`，在 Windows 可去掉该参数。
- 缺少模型：请按“模型文件”一节准备对应目录/文件。

## 许可证

仅供学习与研究使用，模型与数据请遵循其各自授权。
