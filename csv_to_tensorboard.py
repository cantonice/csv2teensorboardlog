# 文件名: csv_to_tensorboard.py
import os
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path

def csv_to_tensorboard(csv_file, output_dir=None, model_name=None):
    """
    将CSV训练日志文件转换为TensorBoard格式
    
    参数:
        csv_file (str): CSV文件路径
        output_dir (str): 输出TensorBoard日志的目录路径
        model_name (str): 模型名称，如果未指定将从文件名获取
    """
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 从文件名提取模型名称(如果未指定)
    if model_name is None:
        model_name = Path(csv_file).stem
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join('runs', 'tensorboard_logs', model_name)
    else:
        output_dir = os.path.join(output_dir, model_name)
        
    # 创建TensorBoard SummaryWriter
    print(f"创建TensorBoard日志到: {output_dir}")
    writer = SummaryWriter(output_dir)
    
    # 提取所有唯一的指标类型
    metric_columns = [col for col in df.columns if col not in ['epoch', 'time']]
    
    # 将每个指标记录到TensorBoard
    for col in metric_columns:
        for idx, row in df.iterrows():
            epoch = row['epoch']
            value = row[col]
            if pd.notnull(value) and not np.isinf(value):  # 跳过NaN和inf值
                writer.add_scalar(col, value, epoch)
    
    # 添加一些聚合指标
    if all(col in df.columns for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
        for idx, row in df.iterrows():
            epoch = row['epoch']
            total_loss = row['train/box_loss'] + row['train/cls_loss'] + row['train/dfl_loss']
            writer.add_scalar('train/total_loss', total_loss, epoch)
    
    if all(col in df.columns for col in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']):
        for idx, row in df.iterrows():
            epoch = row['epoch']
            val_total_loss = row['val/box_loss'] + row['val/cls_loss'] + row['val/dfl_loss']
            writer.add_scalar('val/total_loss', val_total_loss, epoch)
    
    # 保存并关闭writer
    writer.close()
    print(f"转换完成! 使用以下命令启动TensorBoard查看结果:")
    print(f"tensorboard --logdir={output_dir}")

def process_multiple_csv(csv_dir, output_dir=None):
    """
    处理目录中的多个CSV文件
    
    参数:
        csv_dir (str): 包含CSV文件的目录
        output_dir (str): 输出TensorBoard日志的目录路径
    """
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"在 {csv_dir} 中未找到CSV文件")
        return
        
    for csv_file in csv_files:
        full_path = os.path.join(csv_dir, csv_file)
        model_name = Path(csv_file).stem
        csv_to_tensorboard(full_path, output_dir, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将CSV训练日志转换为TensorBoard格式')
    parser.add_argument('--csv_file', type=str, help='单个CSV文件的路径')
    parser.add_argument('--csv_dir', type=str, help='包含多个CSV文件的目录')
    parser.add_argument('--output_dir', type=str, default='runs/tensorboard_logs', help='输出TensorBoard日志的目录')
    parser.add_argument('--model_name', type=str, help='模型名称(仅用于单个文件)')
    
    args = parser.parse_args()
    
    if args.csv_file:
        csv_to_tensorboard(args.csv_file, args.output_dir, args.model_name)
    elif args.csv_dir:
        process_multiple_csv(args.csv_dir, args.output_dir)
    else:
        print("请提供 --csv_file 或 --csv_dir 参数")