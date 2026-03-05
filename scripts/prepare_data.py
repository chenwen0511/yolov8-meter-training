#!/usr/bin/env python3
"""
数据集划分和准备脚本
将增强后的数据划分为训练集和验证集
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List


def split_dataset(
    source_images_dir: str,
    source_labels_dir: str,
    output_dir: str = "data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    划分数据集为训练集和验证集
    
    Args:
        source_images_dir: 源图片目录
        source_labels_dir: 源标注目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        (训练集文件列表, 验证集文件列表)
    """
    
    random.seed(seed)
    
    # 获取所有图片文件
    image_files = sorted([f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        raise ValueError(f"在 {source_images_dir} 中没有找到图片文件")
    
    print(f"找到 {len(image_files)} 个样本")
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 划分
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    
    print(f"训练集: {len(train_files)} 个样本")
    print(f"验证集: {len(val_files)} 个样本")
    
    # 创建目录结构
    splits = ['train', 'val']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # 复制文件
    def copy_files(files: List[str], split: str):
        for img_file in files:
            # 复制图片
            src_img = os.path.join(source_images_dir, img_file)
            dst_img = os.path.join(output_dir, split, 'images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # 复制标注
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            src_label = os.path.join(source_labels_dir, label_file)
            dst_label = os.path.join(output_dir, split, 'labels', label_file)
            
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告: 找不到标注文件 {src_label}")
    
    print(f"\n复制训练集文件...")
    copy_files(train_files, 'train')
    
    print(f"复制验证集文件...")
    copy_files(val_files, 'val')
    
    print(f"\n数据集划分完成! 输出目录: {output_dir}")
    
    return train_files, val_files


def create_data_yaml(
    data_dir: str = "data",
    output_path: str = "configs/data.yaml",
    class_names: List[str] = None
):
    """
    创建YOLO数据配置文件
    
    Args:
        data_dir: 数据根目录
        output_path: 输出配置文件路径
        class_names: 类别名称列表
    """
    
    if class_names is None:
        class_names = ["electricity meter"]
    
    # 获取绝对路径
    data_dir_abs = os.path.abspath(data_dir)
    
    config = {
        "path": data_dir_abs,
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入YAML
    with open(output_path, 'w') as f:
        yaml_str = f"""# YOLOv8 数据配置文件
path: {config['path']}  # 数据根目录

# 训练和验证图片路径
train: {config['train']}
val: {config['val']}

# 类别数量
nc: {config['nc']}

# 类别名称
names:
"""
        for i, name in enumerate(class_names):
            yaml_str += f"  {i}: {name}\n"
    
    print(f"数据配置文件已创建: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="划分YOLO数据集")
    parser.add_argument("--images", type=str, required=True, help="源图片目录")
    parser.add_argument("--labels", type=str, required=True, help="源标注目录")
    parser.add_argument("--output", type=str, default="data", help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 划分数据集
    train_files, val_files = split_dataset(
        source_images_dir=args.images,
        source_labels_dir=args.labels,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # 创建数据配置文件
    create_data_yaml(
        data_dir=args.output,
        output_path="configs/data.yaml"
    )
