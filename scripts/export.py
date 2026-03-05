#!/usr/bin/env python3
"""
模型导出脚本 - 将YOLOv8模型导出为其他格式
支持: ONNX, TensorRT, OpenVINO, CoreML, etc.
"""

import os
import argparse
from ultralytics import YOLO


def export_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True
):
    """
    导出YOLOv8模型
    
    Args:
        model_path: 模型文件路径
        format: 导出格式 (onnx, engine, openvino, coreml, etc.)
        imgsz: 输入图像尺寸
        half: 是否使用FP16半精度
        simplify: 是否简化ONNX模型
    """
    
    # 加载模型
    model = YOLO(model_path)
    
    print("=" * 60)
    print(f"导出模型: {model_path}")
    print(f"目标格式: {format}")
    print(f"图像尺寸: {imgsz}")
    print(f"半精度: {half}")
    print("=" * 60)
    
    # 导出
    model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        simplify=simplify
    )
    
    print("\n导出完成!")
    
    # 显示导出文件路径
    export_path = model_path.replace('.pt', f'.{format}')
    if format == "engine":
        export_path = model_path.replace('.pt', '.engine')
    elif format == "openvino":
        export_path = model_path.replace('.pt', '_openvino_model')
    
    print(f"导出文件: {export_path}")
    
    return export_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出YOLOv8模型")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--format", type=str, default="onnx", 
                       choices=["onnx", "engine", "openvino", "coreml", "tflite", "pb", "torchscript"],
                       help="导出格式")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--half", action="store_true", help="使用FP16半精度")
    parser.add_argument("--simplify", action="store_true", default=True, help="简化ONNX模型")
    
    args = parser.parse_args()
    
    export_model(
        model_path=args.model,
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        simplify=args.simplify
    )
