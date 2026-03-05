#!/usr/bin/env python3
"""
YOLOv8 电表检测训练脚本 - 完整版
包含：数据准备、模型训练、验证和ONNX导出
"""

import os
import yaml
import shutil
import random
import warnings
from pathlib import Path
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# 配置
AUGMENTED_DATA_DIR = "/root/.openclaw/workspace/meter_augmented"
WORKSPACE_DIR = "/root/.openclaw/workspace/yolov8-meter-training"
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
CONFIGS_DIR = os.path.join(WORKSPACE_DIR, "configs")

def prepare_dataset():
    """划分数据集为训练集和验证集 (80/20)"""
    print("="*60)
    print("步骤 1: 准备数据集")
    print("="*60)
    
    source_images = os.path.join(AUGMENTED_DATA_DIR, "images")
    source_labels = os.path.join(AUGMENTED_DATA_DIR, "labels")
    
    # 获取所有图片
    image_files = sorted([f for f in os.listdir(source_images) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        raise ValueError(f"在 {source_images} 中没有找到图片文件")
    
    print(f"找到 {len(image_files)} 个样本")
    
    # 随机打乱并划分
    random.seed(42)
    random.shuffle(image_files)
    
    n_total = len(image_files)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    
    print(f"训练集: {len(train_files)} 个样本 ({len(train_files)/n_total*100:.1f}%)")
    print(f"验证集: {len(val_files)} 个样本 ({len(val_files)/n_total*100:.1f}%)")
    
    # 创建目录结构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATA_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, split, 'labels'), exist_ok=True)
    
    # 复制文件
    def copy_split(files, split_name):
        for img_file in files:
            # 复制图片
            src_img = os.path.join(source_images, img_file)
            dst_img = os.path.join(DATA_DIR, split_name, 'images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # 复制标注
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            src_label = os.path.join(source_labels, label_file)
            dst_label = os.path.join(DATA_DIR, split_name, 'labels', label_file)
            
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    print(f"\n复制训练集文件...")
    copy_split(train_files, 'train')
    
    print(f"复制验证集文件...")
    copy_split(val_files, 'val')
    
    # 创建数据配置文件
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    data_config = {
        'path': DATA_DIR,
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': {0: 'electricity_meter'}
    }
    
    yaml_path = os.path.join(CONFIGS_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n数据配置文件已创建: {yaml_path}")
    return yaml_path

def train_model(data_yaml_path):
    """训练YOLOv8n模型 100 epochs"""
    print("\n" + "="*60)
    print("步骤 2: 训练模型")
    print("="*60)
    
    # 加载预训练模型
    print("加载 YOLOv8n 预训练模型...")
    model = YOLO('yolov8n.pt')
    
    # 训练参数
    train_args = {
        'data': data_yaml_path,
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': 'cpu',
        'project': MODELS_DIR,
        'name': 'meter_detection',
        'exist_ok': True,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'verbose': True,
    }
    
    print(f"\n训练配置:")
    print(f"  - 模型: yolov8n")
    print(f"  - 训练轮数: 100")
    print(f"  - 图像尺寸: 640")
    print(f"  - 批量大小: 8")
    print(f"  - 训练设备: cpu")
    print(f"  - 早停耐心: 20 epochs")
    
    print("\n开始训练...")
    results = model.train(**train_args)
    
    return model, results

def validate_model(model):
    """验证模型并输出指标"""
    print("\n" + "="*60)
    print("步骤 3: 验证模型")
    print("="*60)
    
    metrics = model.val()
    
    print(f"\n验证结果:")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - mAP75: {metrics.box.map75:.4f}")
    
    return metrics

def export_onnx(model):
    """导出模型为ONNX格式"""
    print("\n" + "="*60)
    print("步骤 4: 导出ONNX模型")
    print("="*60)
    
    model_path = os.path.join(MODELS_DIR, 'meter_detection', 'weights', 'best.pt')
    
    # 加载最佳模型
    best_model = YOLO(model_path)
    
    # 导出ONNX
    print("导出为ONNX格式...")
    best_model.export(format='onnx', imgsz=640, simplify=True)
    
    onnx_path = model_path.replace('.pt', '.onnx')
    
    if os.path.exists(onnx_path):
        print(f"ONNX模型已保存: {onnx_path}")
        
        # 复制到models根目录
        final_onnx = os.path.join(MODELS_DIR, 'meter_detection.onnx')
        shutil.copy2(onnx_path, final_onnx)
        print(f"ONNX模型已复制到: {final_onnx}")
        return final_onnx
    else:
        print("警告: ONNX导出可能失败")
        return None

def main():
    """主函数"""
    print("\n" + "="*70)
    print(" YOLOv8 电表检测模型训练")
    print("="*70)
    
    try:
        # 步骤1: 准备数据
        data_yaml = prepare_dataset()
        
        # 步骤2: 训练模型
        model, results = train_model(data_yaml)
        
        # 步骤3: 验证模型
        metrics = validate_model(model)
        
        # 步骤4: 导出ONNX
        onnx_path = export_onnx(model)
        
        # 输出最终结果
        print("\n" + "="*70)
        print(" 训练完成!")
        print("="*70)
        print(f"\n模型文件位置:")
        print(f"  - 最佳模型: {MODELS_DIR}/meter_detection/weights/best.pt")
        print(f"  - 最后模型: {MODELS_DIR}/meter_detection/weights/last.pt")
        if onnx_path:
            print(f"  - ONNX模型: {onnx_path}")
        
        print(f"\n最终指标:")
        print(f"  - mAP50: {metrics.box.map50:.4f}")
        print(f"  - mAP50-95: {metrics.box.map:.4f}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        raise

if __name__ == "__main__":
    main()
