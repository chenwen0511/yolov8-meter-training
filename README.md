# YOLOv8 电表检测训练项目

基于 YOLOv8 的电表检测模型训练、导出和推理完整方案。

## 项目结构

```
yolov8-meter-training/
├── configs/              # 配置文件
│   └── data.yaml         # 数据集配置
├── data/                 # 数据集 (需要自行准备)
│   ├── train/
│   │   ├── images/       # 训练图片
│   │   └── labels/       # 训练标注
│   └── val/
│       ├── images/       # 验证图片
│       └── labels/       # 验证标注
├── models/               # 训练输出目录
│   └── meter_detection/  # 训练结果
│       ├── weights/
│       │   ├── best.pt   # 最佳模型
│       │   └── last.pt   # 最后模型
│       └── results.png   # 训练曲线
├── scripts/              # 脚本文件
│   ├── prepare_data.py   # 数据划分
│   ├── train.py          # 训练脚本
│   ├── detect.py         # 推理脚本 (PyTorch)
│   ├── detect_onnx.py    # 推理脚本 (ONNX)
│   └── export.py         # 模型导出
├── requirements.txt      # 依赖列表
└── README.md             # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将生成的增强数据放入 `data/` 目录：

```bash
# 划分训练集和验证集
python scripts/prepare_data.py \
    --images /path/to/meter_augmented/images \
    --labels /path/to/meter_augmented/labels \
    --output data \
    --train-ratio 0.8
```

### 3. 训练模型

```bash
# 使用 YOLOv8n (nano) 模型训练
python scripts/train.py \
    --model yolov8n \
    --epochs 100 \
    --batch 16 \
    --device 0

# 或使用其他模型尺寸
# yolov8n (最快), yolov8s, yolov8m, yolov8l, yolov8x (最准确)
```

### 4. 导出模型

```bash
# 导出为 ONNX 格式
python scripts/export.py \
    --model models/meter_detection/weights/best.pt \
    --format onnx

# 导出为 TensorRT (GPU加速)
python scripts/export.py \
    --model models/meter_detection/weights/best.pt \
    --format engine \
    --half
```

### 5. 运行推理

**使用 PyTorch 模型:**
```bash
python scripts/detect.py \
    --model models/meter_detection/weights/best.pt \
    --source path/to/image.jpg \
    --save output.jpg \
    --show
```

**使用 ONNX 模型 (无需 PyTorch):**
```bash
python scripts/detect_onnx.py \
    --model models/meter_detection/weights/best.onnx \
    --source path/to/image.jpg \
    --save output.jpg \
    --show
```

## 训练参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| --model | yolov8n | 模型尺寸 (n/s/m/l/x) |
| --epochs | 100 | 训练轮数 |
| --imgsz | 640 | 输入图像尺寸 |
| --batch | 16 | 批量大小 |
| --device | 0 | GPU设备号，CPU用 "cpu" |
| --data | configs/data.yaml | 数据配置文件 |

## 模型性能对比

| 模型 | 参数量 | mAP@50 | 推理速度 (CPU) | 推理速度 (GPU) |
|-----|--------|--------|---------------|---------------|
| yolov8n | 3.2M | 待训练 | ~10ms | ~2ms |
| yolov8s | 11.2M | 待训练 | ~20ms | ~3ms |
| yolov8m | 25.9M | 待训练 | ~40ms | ~5ms |
| yolov8l | 43.7M | 待训练 | ~70ms | ~8ms |
| yolov8x | 68.2M | 待训练 | ~120ms | ~12ms |

## 导出格式支持

- `onnx` - ONNX (跨平台)
- `engine` - TensorRT (NVIDIA GPU加速)
- `openvino` - OpenVINO (Intel CPU优化)
- `coreml` - CoreML (Apple设备)
- `tflite` - TFLite (移动端)

## 数据增强建议

由于我们使用合成数据增强，建议在真实场景数据上验证模型性能。如果检测效果不佳，可以：

1. 收集更多真实场景图片
2. 使用 Grounding DINO 进行预标注
3. 人工修正标注后重新训练
4. 添加更多数据增强（颜色抖动、随机噪声等）

## 许可证

MIT License
