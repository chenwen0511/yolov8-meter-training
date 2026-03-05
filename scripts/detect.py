#!/usr/bin/env python3
"""
YOLOv8 电表检测推理脚本
"""

import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


def detect_image(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.25,
    save_path: str = None,
    show: bool = False
):
    """
    对单张图片进行电表检测
    
    Args:
        model_path: 模型文件路径
        image_path: 图片文件路径
        conf_threshold: 置信度阈值
        save_path: 保存路径
        show: 是否显示结果
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 推理
    results = model(image_path, conf=conf_threshold)
    
    # 处理结果
    for result in results:
        boxes = result.boxes
        
        # 打印检测结果
        print(f"\n检测到 {len(boxes)} 个电表:")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            print(f"  [{i+1}] 置信度: {conf:.3f}, 框: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # 保存结果
        if save_path:
            result.save(filename=save_path)
            print(f"\n结果已保存: {save_path}")
        
        # 显示结果
        if show:
            result.show()
    
    return results


def detect_video(
    model_path: str,
    video_path: str,
    conf_threshold: float = 0.25,
    output_path: str = None,
    show: bool = False
):
    """
    对视频进行电表检测
    
    Args:
        model_path: 模型文件路径
        video_path: 视频文件路径 (或用0表示摄像头)
        conf_threshold: 置信度阈值
        output_path: 输出视频路径
        show: 是否显示实时画面
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 打开视频
    if video_path == "0":
        cap = cv2.VideoCapture(0)
        print("使用摄像头")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"打开视频: {video_path}")
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"输出视频: {output_path}")
    
    frame_count = 0
    
    print("开始检测... (按 'q' 退出)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # 绘制结果
        annotated_frame = results[0].plot()
        
        # 保存帧
        if writer:
            writer.write(annotated_frame)
        
        # 显示
        if show:
            cv2.imshow('Meter Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")
    
    # 释放资源
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n完成! 共处理 {frame_count} 帧")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8电表检测")
    parser.add_argument("--model", type=str, default="models/meter_detection/weights/best.pt", help="模型路径")
    parser.add_argument("--source", type=str, required=True, help="图片/视频路径 (或0表示摄像头)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--save", type=str, default=None, help="保存路径")
    parser.add_argument("--show", action="store_true", help="是否显示结果")
    
    args = parser.parse_args()
    
    # 判断是图片还是视频
    if args.source == "0" or args.source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detect_video(
            model_path=args.model,
            video_path=args.source,
            conf_threshold=args.conf,
            output_path=args.save,
            show=args.show
        )
    else:
        detect_image(
            model_path=args.model,
            image_path=args.source,
            conf_threshold=args.conf,
            save_path=args.save,
            show=args.show
        )
