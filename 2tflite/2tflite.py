import tensorflow as tf
import torch
import os
import sys
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
# 获取当前文件所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到系统路径
sys.path.append(parent_dir)

# 现在你可以导入file2.py中的类
from public.dataset import ECGDataset



DATA_DIR = '../test_data'
BATCH_SIZE = 64
NUM_WORKERS = 4
PREFETCH_FACTOR = 2

dataset = ECGDataset(DATA_DIR)
indices = list(range(50000))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                          prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义代表性数据集函数
def representative_data_gen():
    for data, labels in tqdm(val_loader, desc="Preparing Representative Dataset"):  # 加入进度条
        data = data.to(device)  # Ensure the input data is on the same device as the model
        data = data.cpu().numpy()  # 将数据从 PyTorch Tensor 转换为 NumPy 数组
        for i in range(data.shape[0]):  # 逐样本处理数据
            yield [data[i:i+1].astype(np.float32)]  # 返回单个样本，并确保数据类型为 float32

saved_model_dir = '/home/jostar/workspace/onnx2tflite/tflite'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # 或 tf.int8
converter.inference_output_type = tf.int8  # 或 tf.int8

# 转换模型
tflite_quant_model = converter.convert()

# 保存量化后的 TFLite 模型
with open('model_quant_integer.tflite', 'wb') as f:
    f.write(tflite_quant_model)
