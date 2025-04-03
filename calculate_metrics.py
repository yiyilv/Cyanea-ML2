#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
计算集成模型的性能指标，包括召回率和F1-Score
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, precision_score

def calculate_ensemble_metrics(predictions_file='models/prediction_results.csv'):
    """
    计算集成模型的性能指标
    
    参数:
        predictions_file: 预测结果CSV文件路径
    """
    # 加载预测结果
    print(f"Loading predictions from: {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    # 确保有真实标签和预测标签
    if 'true_mode' not in df.columns or 'predicted_mode' not in df.columns:
        print("Error: Missing required columns 'true_mode' or 'predicted_mode'")
        return
    
    # 提取真实标签和预测标签
    y_true = df['true_mode'].values
    y_pred = df['predicted_mode'].values
    
    # 计算整体准确率
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {accuracy:.4f} ({(accuracy*100):.2f}%)")
    
    # 计算按类别的指标
    class_names = ['Watching Videos', 'Browsing Infinite Feed', 'Other Activities']
    
    # 计算精确率、召回率和F1-Score
    precision = precision_score(y_true, y_pred, average=None, labels=[1, 2, 3])
    recall = recall_score(y_true, y_pred, average=None, labels=[1, 2, 3])
    f1 = f1_score(y_true, y_pred, average=None, labels=[1, 2, 3])
    
    # 打印按类别的指标
    print("\nMetrics by class:")
    print("-" * 80)
    print(f"{'Class':<25} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision[i]:<15.4f} {recall[i]:<15.4f} {f1[i]:<15.4f}")
    
    # 计算宏平均和加权平均
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("-" * 80)
    print(f"{'Macro average':<25} {macro_precision:<15.4f} {macro_recall:<15.4f} {macro_f1:<15.4f}")
    print(f"{'Weighted average':<25} {weighted_precision:<15.4f} {weighted_recall:<15.4f} {weighted_f1:<15.4f}")
    
    # 打印完整的分类报告
    print("\nDetailed Classification Report:")
    print("-" * 80)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    print("\nConfusion Matrix:")
    print("-" * 80)
    print("                       Predicted")
    print("                       Watching Videos  Browsing Infinite Feed  Other Activities")
    print("-" * 80)
    for i, class_name in enumerate(class_names):
        print(f"True {class_name:<15} {cm[i][0]:<18} {cm[i][1]:<24} {cm[i][2]:<15}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate metrics for ensemble model")
    parser.add_argument("--predictions", type=str, default="models/prediction_results.csv", 
                        help="Path to predictions CSV file")
    
    args = parser.parse_args()
    
    calculate_ensemble_metrics(args.predictions) 