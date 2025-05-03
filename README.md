# CSV 转 TensorBoard 日志文件 脚本

这个工具可以将 YOLO 训练日志的 CSV 文件转换为 TensorBoard 日志文件，方便直观比较不同模型的训练过程和性能指标。

## 功能特点

- 将单个或多个 CSV 训练日志文件转换为 TensorBoard 日志文件
- 递归搜索：自动在指定目录及其所有子目录中查找 CSV 文件
- 保留目录结构：在 TensorBoard 中保持原始的目录组织结构
- 自动处理无效值 (NaN, inf)
- 计算并添加聚合指标（如总损失）
- 支持批量处理目录中的所有 CSV 文件
- 自定义输出目录和模型名称

## 安装依赖

```bash
pip install pandas numpy torch tensorboard
```

## 使用方法

### 转换单个 CSV 文件

```bash
python csv_to_tensorboard.py --csv_file /path/to/yolov12n.csv
```

### 转换目录中的所有 CSV 文件

```bash
python csv_to_tensorboard.py --csv_dir /home/coderzhou/github_project/new_yolov12/yolov12/logs
```

### 指定输出目录

```bash
python csv_to_tensorboard.py --csv_file /path/to/yolov12n.csv --output_dir /path/to/output
```

### 指定模型名称

```bash
python csv_to_tensorboard.py --csv_file /path/to/model_log.csv --model_name yolov12-custom
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--csv_file` | 单个 CSV 文件的路径 |
| `--csv_dir` | 包含多个 CSV 文件的目录（将递归搜索所有子目录） |
| `--output_dir` | 输出 TensorBoard 日志的目录 (默认: `runs/tensorboard_logs`) |
| `--model_name` | 模型名称 (仅用于单个文件，默认使用 CSV 文件名) |

## 查看结果

转换完成后，通过以下命令启动 TensorBoard 查看可视化结果:

```bash
tensorboard --logdir=runs/tensorboard_logs
```

然后在浏览器中访问 `http://localhost:6006` 查看结果。

## 在 TensorBoard 中可以查看的内容

- 不同模型的训练和验证损失对比
- 精度 (Precision)、召回率 (Recall) 和 mAP 指标
- 学习率变化趋势
- 自定义聚合指标 (如总损失)

## 示例

### 转换单个模型日志

```bash
python csv_to_tensorboard.py --csv_file /home/coderzhou/github_project/new_yolov12/yolov12/logs/yolov12n.csv
```

### 转换并比较多个模型

```bash
python csv_to_tensorboard.py --csv_dir /home/coderzhou/github_project/new_yolov12/yolov12/logs
```

## 注意事项

- CSV 文件应包含表头，并至少包含 `epoch` 列
- 对于包含 NaN 或 inf 值的指标将被自动跳过
- 确保 TensorBoard 安装正确，可通过 `pip install tensorboard` 安装