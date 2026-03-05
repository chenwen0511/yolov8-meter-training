#!/bin/bash
# 快速开始脚本 - 一键训练电表检测模型

set -e

echo "=============================================="
echo "YOLOv8 电表检测 - 快速开始"
echo "=============================================="

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <images_dir> <labels_dir> [model_size]"
    echo ""
    echo "参数:"
    echo "  images_dir   - 增强后的图片目录"
    echo "  labels_dir   - YOLO标注文件目录"
    echo "  model_size   - 模型尺寸 (n/s/m/l/x), 默认: n"
    echo ""
    echo "示例:"
    echo "  $0 ../meter_augmented/images ../meter_augmented/labels n"
    exit 1
fi

IMAGES_DIR=$1
LABELS_DIR=$2
MODEL_SIZE=${3:-"n"}

echo ""
echo "配置:"
echo "  图片目录: $IMAGES_DIR"
echo "  标注目录: $LABELS_DIR"
echo "  模型尺寸: yolov8${MODEL_SIZE}"
echo ""

# 检查目录是否存在
if [ ! -d "$IMAGES_DIR" ]; then
    echo "错误: 图片目录不存在: $IMAGES_DIR"
    exit 1
fi

if [ ! -d "$LABELS_DIR" ]; then
    echo "错误: 标注目录不存在: $LABELS_DIR"
    exit 1
fi

# 步骤1: 数据划分
echo "[1/3] 划分训练集和验证集..."
python scripts/prepare_data.py \
    --images "$IMAGES_DIR" \
    --labels "$LABELS_DIR" \
    --output data \
    --train-ratio 0.8

# 步骤2: 训练模型
echo ""
echo "[2/3] 训练 YOLOv8${MODEL_SIZE} 模型..."
python scripts/train.py \
    --model "yolov8${MODEL_SIZE}" \
    --epochs 100 \
    --batch 16 \
    --data configs/data.yaml

# 步骤3: 导出ONNX模型
echo ""
echo "[3/3] 导出 ONNX 模型..."
python scripts/export.py \
    --model "models/meter_detection/weights/best.pt" \
    --format onnx

echo ""
echo "=============================================="
echo "训练完成!"
echo "=============================================="
echo ""
echo "输出文件:"
echo "  PyTorch模型: models/meter_detection/weights/best.pt"
echo "  ONNX模型:    models/meter_detection/weights/best.onnx"
echo ""
echo "使用模型进行推理:"
echo "  python scripts/detect.py --model models/meter_detection/weights/best.pt --source your_image.jpg"
echo ""
