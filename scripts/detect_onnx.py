#!/usr/bin/env python3
"""
ONNX 推理脚本 - 无需 PyTorch 环境
使用 ONNX Runtime 进行电表检测
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import argparse


class MeterDetectorONNX:
    """ONNX 电表检测器"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化检测器
        
        Args:
            model_path: ONNX模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 创建ONNX Runtime会话
        providers = ort.get_available_providers()
        print(f"可用 providers: {providers}")
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2]  # 通常是 640
        
        print(f"模型输入尺寸: {self.input_shape}")
        
        # 类别名称
        self.class_names = ["electricity meter"]
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图片"""
        # 调整尺寸
        img = cv2.resize(image, (self.img_size, self.img_size))
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 添加batch维度
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess(self, outputs: np.ndarray, orig_shape: tuple) -> list:
        """后处理检测结果"""
        # YOLOv8输出格式: [batch, 84, 8400] (x, y, w, h, conf, cls1, cls2...)
        predictions = outputs[0]  # [84, 8400]
        
        # 转置: [8400, 84]
        predictions = np.transpose(predictions, (1, 0))
        
        boxes = []
        scores = []
        class_ids = []
        
        for pred in predictions:
            # 获取类别置信度
            class_conf = pred[4:]
            class_id = np.argmax(class_conf)
            confidence = class_conf[class_id]
            
            if confidence < self.conf_threshold:
                continue
            
            # 获取框坐标 (中心点 + 宽高)
            cx, cy, w, h = pred[:4]
            
            # 转换为左上角坐标
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            class_ids.append(class_id)
        
        # NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            
            results = []
            for i in indices:
                box = boxes[i]
                # 将坐标从640x640映射回原图尺寸
                x1, y1, x2, y2 = box
                x1 = int(x1 * orig_shape[1] / self.img_size)
                y1 = int(y1 * orig_shape[0] / self.img_size)
                x2 = int(x2 * orig_shape[1] / self.img_size)
                y2 = int(y2 * orig_shape[0] / self.img_size)
                
                results.append({
                    'box': [x1, y1, x2, y2],
                    'score': float(scores[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': self.class_names[int(class_ids[i])]
                })
            
            return results
        
        return []
    
    def detect(self, image: np.ndarray) -> list:
        """
        检测电表
        
        Args:
            image: BGR格式的numpy数组
        
        Returns:
            检测结果列表
        """
        orig_shape = image.shape[:2]
        
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # 后处理
        results = self.postprocess(outputs, orig_shape)
        
        return results
    
    def draw_detections(self, image: np.ndarray, results: list) -> np.ndarray:
        """绘制检测结果"""
        img = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['box']
            score = result['score']
            class_name = result['class_name']
            
            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 画标签
            label = f"{class_name}: {score:.3f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return img


def main():
    parser = argparse.ArgumentParser(description="ONNX电表检测")
    parser.add_argument("--model", type=str, required=True, help="ONNX模型路径")
    parser.add_argument("--source", type=str, required=True, help="图片路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--save", type=str, default=None, help="保存路径")
    parser.add_argument("--show", action="store_true", help="显示结果")
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = MeterDetectorONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # 读取图片
    image = cv2.imread(args.source)
    if image is None:
        print(f"无法读取图片: {args.source}")
        return
    
    # 检测
    results = detector.detect(image)
    
    print(f"\n检测到 {len(results)} 个电表:")
    for i, result in enumerate(results):
        print(f"  [{i+1}] {result['class_name']}: {result['score']:.3f}")
        print(f"      框: {result['box']}")
    
    # 绘制结果
    output = detector.draw_detections(image, results)
    
    # 保存
    if args.save:
        cv2.imwrite(args.save, output)
        print(f"\n结果已保存: {args.save}")
    
    # 显示
    if args.show:
        cv2.imshow("Meter Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
