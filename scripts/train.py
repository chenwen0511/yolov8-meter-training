#!/usr/bin/env python3
"""
YOLOv8 电表检测训练脚本
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def train_model(
    data_yaml_path: str = "configs/data.yaml",
    model_size: str = "yolov8n",  # n/s/m/l/x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",  # GPU设备号，CPU用 "cpu"
    project: str = "models",
    name: str = "meter_detection"
):
    """
    训练YOLOv8模型
    
    Args:
        data_yaml_path: 数据配置文件路径
        model_size: 模型大小 (n/s/m/l/x)
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: 批量大小
        device: 训练设备
        project: 输出项目目录
        name: 训练名称
    """
    
    # 加载预训练模型
    model = YOLO(f"{model_size}.pt")
    
    # 训练参数
    train_args = {
        "data": data_yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
        "exist_ok": True,
        "patience": 20,  # 早停耐心值
        "save": True,
        "save_period": 10,  # 每10轮保存一次
        "verbose": True,
    }
    
    print("=" * 60)
    print("开始训练 YOLOv8 电表检测模型")
    print("=" * 60)
    print(f"模型: {model_size}")
    print(f"数据配置: {data_yaml_path}")
    print(f"训练轮数: {epochs}")
    print(f"图像尺寸: {imgsz}")
    print(f"批量大小: {batch}")
    print(f"训练设备: {device}")
    print("=" * 60)
    
    # 开始训练
    results = model.train(**train_args)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳模型: {project}/{name}/weights/best.pt")
    print(f"最后模型: {project}/{name}/weights/last.pt")
    
    # 验证模型
    print("\n验证模型...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练YOLOv8电表检测模型")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="数据配置文件路径")
    parser.add_argument("--model", type=str, default="yolov8n", choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"], help="模型大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批量大小")
    parser.add_argument("--device", type=str, default="0", help="训练设备 (GPU号或cpu)")
    parser.add_argument("--project", type=str, default="models", help="输出目录")
    parser.add_argument("--name", type=str, default="meter_detection", help="训练名称")
    
    args = parser.parse_args()
    
    train_model(
        data_yaml_path=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )
