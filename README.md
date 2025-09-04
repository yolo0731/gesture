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

（模块运行方式已合并至 run.py，推荐直接使用 run.py）

- 窗口内按钮功能：
  - 1. 手写数字识别：空中书写数字，实时识别与显示
  - 2. 手写字母识别：空中书写字母，实时识别与显示
  - 3. 数字手势识别：对准摄像头做 ASL 数字手势
  - 4. 字母手势识别：对准摄像头做 ASL 字母手势
  - 5. 动态手势识别：Click/Stop/Rotate/No
  - 6. 动态字母手势：J/Z（轨迹识别）
  - 7. 手指指向识别：食指指尖方向判定（上/下/左/右），按 q 退出

### 手指指向识别（新功能）

- 基于 MediaPipe Hands 关键点，无需训练模型；通过食指 PIP(6)→TIP(8) 向量判断方向，并用滑动窗口去抖动。
- 使用方式：
  - 交互界面：点击“7.手指指向识别”。对准摄像头伸出食指，轻微改变指尖朝向，上/下/左/右将叠加显示，并写入右侧“识别结果”。按键盘 `q` 退出。
  - Demo：`PYTHONPATH=src python -m demos.direction_gesture`
  - 摄像头索引异常时，先运行：`python scripts/detect_and_update_camera_index.py --dry-run`

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
PYTHONPATH=src python -m ml.cnntrain_mnist

# 训练完成后会自动保存到 models/MNIST1.pth
```

2) 训练并生成字母识别模型（EMNIST Letters，PyTorch）

```
conda activate gesture
PYTHONPATH=src python -m ml.duan_validation_2

# 训练完成后会自动保存到 models/EMNIST2_5.18.pth（state_dict）
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

## 目录说明（src 布局）

- 源代码：`src/`
  - UI：`ui/main_window.py`（原 `jiemianshixian.py`），`ui/resources/interaction.ui`，`ui/interaction.py`
  - 识别：`recognition/`（手写与静态/动态识别模块）
  - 跟踪：`tracking/`（Mediapipe 封装与结构体）
  - 机器学习：`ml/`（训练脚本与动态手势预测）
  - 工具：`utils/paths.py`
  - 演示：`demos/`（方向/关键点）
- 入口：`run.py`（会自动引入 `src/` 并启动 UI）
- 脚本与工具：`scripts/`（摄像头检测、采集、训练）
- 模型：`models/`（权重与 SavedModel 目录）
- 数据：`data/`（由脚本自动下载/缓存）

数据集目录默认忽略：`data/`（含下载缓存等）。`models/` 现已纳入版本控制，便于开箱即用。

### 项目目录树

```
gesture/
├── README.md
├── requirements.txt
├── environment.yml
├── environment.gpu.yml
├── run.py
│
├── src/
│   ├── （入口已合并到 run.py）
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── interaction.py
│   │   └── resources/
│   │       └── interaction.ui
│   │
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── written_number.py
│   │   ├── written_letter.py
│   │   ├── number_gesture.py
│   │   ├── letter_gesture.py
│   │   ├── direction_gesture.py
│   │   ├── number_rec.py
│   │   ├── letter_rec.py
│   │   └── letter_interpre.py
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── HandTrackingModule.py
│   │   ├── HandTrackingModule_letter.py
│   │   ├── Finger.py
│   │   ├── Landmark.py
│   │   └── definition.py
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── cnntrain_mnist.py
│   │   ├── duan_validation_2.py
│   │   └── dg_prediction_CSRN.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── paths.py
│   │
│   └── demos/
│       ├── __init__.py
│       ├── direction_gesture.py
│       └── hand_keypoints.py
│
├── scripts/
│   ├── collect_sequences.py
│   ├── detect_and_update_camera_index.py
│   ├── test_camera.py
│   └── train_sequence_classifier.py
│
├── data/
│   ├── MNIST/
│   └── EMNIST/
│
├── models/
│   ├── MNIST1.pth
│   ├── EMNIST2_5.18.pth
│   ├── action.h5_3/
│   └── CSRN_Model_2/
│
├── paper/
│   └── 基于视觉信息的手势识别研究(1).pdf
│
└── 测试文件
    ├── test.jpg
    ├── test.txt
    ├── test_letter.jpg
    └── test_letter.txt
```

### 目录内容说明

- 顶层
  - `run.py`: 程序主入口，自动将 `src/` 加入 `sys.path` 并启动 UI。
  - `requirements.txt` / `environment*.yml`: 运行与开发环境依赖清单。
  - `README.md`: 使用说明、训练与目录结构文档。

- `src/`（源码根目录）
  - 入口逻辑已在 `run.py` 中实现，`src/` 下不再单独提供 `app_main.py`。
  - `ui/`（界面层）
    - `main_window.py`: 主窗口逻辑（原 `jiemianshixian.py`），负责加载 `.ui`、绑定按钮到功能函数。
    - `interaction.py`: 由 Qt 生成的 UI Python 文件（可选，默认直接加载 `.ui`）。
    - `resources/interaction.ui`: Qt Designer 生成的界面文件，推荐以此为准编辑 UI。
  - `recognition/`（识别模块）
    - `written_number.py` / `written_letter.py`: 空中手写数字/字母识别逻辑（含摄像头绘制、预处理与模型调用）。
    - `number_gesture.py` / `letter_gesture.py`: 静态数字/字母手势识别（Mediapipe 关键点规则或序列模型）。
    - `direction_gesture.py`: 食指方向（上/下/左/右）识别，基于 6→8 向量与滑动窗口稳定。
    - `number_rec.py` / `letter_rec.py`: 统一的 UI 回调入口，调用对应的 `written_*`。
    - `letter_interpre.py`: 字母手势解释器（将 21 个关键点状态映射到字母类别）。
  - `tracking/`（手部跟踪与结构）
    - `HandTrackingModule*.py`: Mediapipe Hands 的简单封装，提供检测、关键点坐标与手指抬起判断。
    - `Finger.py` / `Landmark.py`: 结构体，存储手指各关节坐标。
    - `definition.py`: 静态数字手势（ASL 数字）的规则映射与辅助函数。
  - `ml/`（训练与模型调用）
    - `cnntrain_mnist.py`: MNIST 数字识别训练脚本，输出 `models/MNIST1.pth`。
    - `duan_validation_2.py`: EMNIST Letters 字母识别训练脚本，输出 `models/EMNIST2_5.18.pth`（state_dict）。
    - `dg_prediction_CSRN.py`: 动态手势（Click/Stop/Rotate/No）序列模型的推理逻辑（加载 TF Keras 模型）。
  - `utils/`
    - `paths.py`: 统一的路径工具，自动定位项目根、`models/`、`data/` 与 UI 资源路径。
  - `demos/`（演示）
    - `hand_keypoints.py`: 实时手部关键点可视化。
    - `direction_gesture.py`: 方向手势 Demo（与 UI 中“手指指向识别”一致）。

### 模式映射（按钮 → 模块）

- 1. 手写数字识别:
  - 入口: `ui/main_window.py` → `recognition.number_rec.number_rec`
  - 实现: `recognition/written_number.py`（摄像头画布、预处理、推理）
  - 模型: `models/MNIST1.pth`
- 2. 手写字母识别:
  - 入口: `ui/main_window.py` → `recognition.letter_rec.letter_rec`
  - 实现: `recognition/written_letter.py`
  - 模型: `models/EMNIST2_5.18.pth`
- 3. 数字手势识别（ASL 数字，静态）:
  - 入口: `ui/main_window.py` → `recognition.number_gesture.number_gesture`
  - 实现: Mediapipe Hands + `tracking/definition.py` 规则映射（无需模型）
- 4. 字母手势识别（ASL 字母，静态）:
  - 入口: `ui/main_window.py` → `recognition.letter_gesture.letter_gesture`
  - 实现: Mediapipe Hands + `recognition/letter_interpre.py`
- 5. 动态手势识别（Click/Stop/Rotate/No）:
  - 入口: `ui/main_window.py` → `ml.dg_prediction_CSRN.CSRN`
  - 模型: `models/CSRN_Model_2/` SavedModel 或 `models/CSRN_Model_2.h5`
- 6. 动态字母手势（J/Z 轨迹）:
  - 入口: `ui/main_window.py` → `recognition.letter_gesture.JZRec`
  - 模型: `models/action.h5_3/` SavedModel 或 `models/action.h5_3.h5`
- 7. 手指指向识别（上/下/左/右）:
  - 入口: `ui/main_window.py` → `recognition.direction_gesture.direction_gesture`
  - 实现: Mediapipe Hands + 6→8 向量方向判定（无需模型）

- `scripts/`（脚本工具）
  - `detect_and_update_camera_index.py`: 自动探测可用摄像头并批量替换代码中的固定索引（支持 `--dry-run`）。
  - `test_camera.py`: 摄像头可用性测试。
  - `collect_sequences.py` / `train_sequence_classifier.py`: 采集 Mediapipe 序列并训练 TF Keras 模型，导出 SavedModel。

- `models/`（模型与权重）
  - `MNIST1.pth`: 手写数字 PyTorch 模型（`written_number.py` 使用）。
  - `EMNIST2_5.18.pth`: 手写字母 PyTorch 模型（`written_letter.py` 使用）。
  - `action.h5_3/`: J/Z 轨迹的 TF Keras SavedModel 目录（含 `saved_model.pb` 与 `variables/`）。
  - `CSRN_Model_2/`: 动态手势四分类的 TF Keras SavedModel 目录。
  - 亦支持 `.h5` 单文件形式：放置为 `models/action.h5_3.h5` 或 `models/CSRN_Model_2.h5`。

- `data/`（数据集缓存）
  - 由 `torchvision.datasets` 或脚本首次运行时自动下载并创建，例如 `data/MNIST/`、`data/EMNIST/`。

- 其他
  - `paper/`: 本地文档（已在 `.gitignore` 忽略）。
  - 测试图片与输出：`test.jpg`、`test_letter.jpg`、`test.txt`、`test_letter.txt`。


附：关键点可视化 Demo：
```
PYTHONPATH=src python -m demos.hand_keypoints
```

附：食指指向识别 Demo：
```
PYTHONPATH=src python -m demos.direction_gesture
```

## 常见问题

- 摄像头打不开或黑屏：请运行 `scripts/detect_and_update_camera_index.py` 自动探测索引，或检查是否被其他应用占用。
- Windows 摄像头后端：部分代码在 Linux 使用 `cv2.CAP_V4L2`，在 Windows 可去掉该参数。
- 缺少模型：请按“模型文件”一节准备对应目录/文件。

## 许可证

仅供学习与研究使用，模型与数据请遵循其各自授权。
