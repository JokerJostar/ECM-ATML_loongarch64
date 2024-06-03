import os

import torch

from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np


# 自定义数据集
class ECGDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        self.labels = [self._get_label(f) for f in self.files]

    def _get_label(self, filename):
        label_part = filename.split('-')[1]
        return 1 if label_part == 'AFIB' else 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.loadtxt(file_path)  # 每行一个数据点
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 增加通道维度和高度维度，形状变为 (1, 1, 1250)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label
