import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import ai_edge_torch
from public.dataset import ECGDataset
from public.model import CustomShuffleNetV2 as Net
import os

def read_txt_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                # 将每行的浮点数读取到列表中，并转换为float32
                numbers = [np.float32(line.strip()) for line in lines]
                data.append(numbers)
    return data

def preprocess_data(data, batch_size):
    # 将数据转换为numpy数组，并转换为float32
    data = np.array(data, dtype=np.float32)
    # 确保数据没有NaN或INF值
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("数据包含NaN或INF值")
    # 确保数据的形状为 (num_samples, 1250)
    data = np.reshape(data, (-1, 1250))
    # 添加额外的维度以匹配目标形状 (batch_size, 1, 1250, 1)
    data = np.expand_dims(data, axis=1)
    data = np.expand_dims(data, axis=-1)
    # 创建TensorFlow数据集
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size)
    return dataset

# 假设你的txt文件在名为'sample_data'的目录中
directoryft = 'test_data'
batch_sizeft = 32

dataft = read_txt_files(directoryft)
datasetft = preprocess_data(dataft, batch_sizeft)

# 加载数据集
DATA_DIR = 'test_data/'
dataset = ECGDataset(DATA_DIR)

# 划分训练集和验证集
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

# 创建数据加载器
BATCH_SIZE = 32
NUM_WORKERS = 4
PREFETCH_FACTOR = 2

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                          prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)

# 定义代表性数据集函数
def representative_dataset():
    for data in datasetft:
        for sample in data:
            # 确保样本的形状与模型输入形状一致
            sample = tf.reshape(sample, [1, 1250, 1])
            yield [sample]

# 加载 PyTorch 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()  # 确保你已经定义了 Net 类
model.load_state_dict=(torch.load('96-96/saved.pth', map_location=device))
model.to(device)
model.eval()

# 获取样本输入
sample_args = next(iter(train_loader))[0]  # 获取一个批次的输入样本

# 将 sample_args 包装在一个元组中
sample_args = (sample_args,)

# 设置量化选项
tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT]}


# 转换并量化模型
edge_model = ai_edge_torch.convert(
    model, sample_args, _ai_edge_converter_flags=tfl_converter_flags
)

# 保存转换后的模型
edge_model.export("af_detection.tflite")