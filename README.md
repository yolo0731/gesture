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

注意：动态/手写识别需要加载本地模型文件。本仓库已随附 `models/` 目录中的预训练权重；数据集将由训练脚本自动下载到 `data/` 目录（无需手动准备）。详见“模型与数据”一节。

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

## 模型与数据

本仓库已经包含 `models/` 目录下的预训练模型，直接运行即可使用；`data/` 目录用于缓存数据集，训练脚本会在首次运行时自动下载与创建（无需手工下载）。

已随仓库提供的模型列表：

- 手写字母轨迹（Keras SavedModel）：`models/action.h5_3/`
- 动态手势（Keras SavedModel）：`models/CSRN_Model_2/`
- 手写数字（PyTorch 权重）：`models/MNIST1.pth`
- 手写字母（PyTorch 权重）：`models/EMNIST2_5.18.pth`

两种使用方式：

1) 使用随附的本地模型（零配置）
- 直接 `python run.py` 启动；界面内相应功能将从 `models/` 加载。无须准备数据集。

2) 自己训练并替换/复现模型（推荐了解流程）
- 见下文“从零开始训练”小节；训练脚本会自动在 `data/` 下下载并缓存 MNIST/EMNIST 等数据集。

### 从零开始训练（自动下载数据）

- 数据集目录 `data/`：由 `torchvision.datasets` 在训练时自动创建并下载，也可手动放置。
- 模型目录 `models/`：用于保存训练得到的权重文件/目录。

推荐使用仓库内训练脚本在本地快速复现（无需提前准备 `data/`）：

1) 训练并生成数字识别模型（MNIST，PyTorch）

```
conda activate gesture  # 或已准备好的 Python 环境
python gesture_app/cnntrain_mnist.py

# 训练完成后会在当前目录生成 MNIST1.pth，将其放入 models/
mkdir -p models
mv MNIST1.pth models/
```

2) 训练并生成字母识别模型（EMNIST Letters，PyTorch）

```
conda activate gesture
python gesture_app/duan_validation_2.py

# 训练完成后会在当前目录生成 EMNIST2.pth（state_dict），
# 本项目推理代码期望文件名为 EMNIST2_5.18.pth，可直接重命名后放入 models/
mkdir -p models
mv EMNIST2.pth models/EMNIST2_5.18.pth
```

运行上述脚本时，`torchvision.datasets` 会自动在本地创建并缓存数据集到 `data/` 目录（首次运行自动下载）：

- `data/MNIST/`（MNIST 数据集）
- `data/EMNIST/`（EMNIST Letters 数据集）

如需手动准备数据集，可参考：

- MNIST：Yann LeCun 官方页面或常见镜像（Kaggle 等），下载后放入 `data/MNIST/`。
- EMNIST Letters：NIST 官方发布页或镜像，下载后放入 `data/EMNIST/`。

3) Keras SavedModel（字母轨迹与动态手势）

- 字母轨迹（J/Z）：`models/action.h5_3/`
- 动态手势（Click/Stop/Rotate/No）：`models/CSRN_Model_2/`

以上两个目录为 Keras SavedModel 目录（包含 `saved_model.pb` 与变量子目录）。你可以按以下方式获取：

- 已有权重：将完整 SavedModel 目录分别拷贝到 `models/action.h5_3/` 与 `models/CSRN_Model_2/`。
- 本地采集+训练（推荐，仓库已提供脚本）：
  - 采集动态手势四分类数据（序列长度 20）：
    ```
    python scripts/collect_sequences.py --labels Click Stop Rotate No --seq-len 20 --samples 80 --out data/CSRN
    ```
  - 训练并导出到 `models/CSRN_Model_2/`：
    ```
    python scripts/train_sequence_classifier.py --data data/CSRN \
      --labels Click Stop Rotate No --seq-len 20 --epochs 30 --batch 32 \
      --out models/CSRN_Model_2
    ```
    注意：标签顺序需与推理代码一致（`Click Stop Rotate No`）。
  - 采集 J/Z/None 三分类数据（序列长度 30）：
    ```
    python scripts/collect_sequences.py --labels Jj Zz None --seq-len 30 --samples 80 --out data/ACTION
    ```
  - 训练并导出到 `models/action.h5_3/`：
    ```
    python scripts/train_sequence_classifier.py --data data/ACTION \
      --labels Jj Zz None --seq-len 30 --epochs 30 --batch 32 \
      --out models/action.h5_3
    ```
    注意：标签顺序需与推理代码一致（`Jj Zz None`）。
- 向维护者获取：若你不方便采集训练，可联系维护者索取已训练好的 SavedModel 目录。

完成以上准备后，`models/` 目录应包含：

- `models/MNIST1.pth`
- `models/EMNIST2_5.18.pth`
- （可选）`models/action.h5_3/` SavedModel 目录
- （可选）`models/CSRN_Model_2/` SavedModel 目录

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
  - `scripts/collect_sequences.py`（采集 Mediapipe 手部关键点序列）
  - `scripts/train_sequence_classifier.py`（训练并导出 Keras SavedModel）

数据集目录默认忽略：`data/`（含下载缓存等）。`models/` 现已纳入版本控制，便于开箱即用。

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
