import torch
import torch.quantization as quantization
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm  # 导入 tqdm 库
import torch
import torch.nn as nn
import sys
import os
from torch.export import export
from executorch.exir import to_edge
from torch.utils.data import Dataset, DataLoader

# 获取当前目录和父目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到系统路径
sys.path.append(parent_dir)

# 从父目录导入模型定义
from public.model import CNNModel
from public.dataset import ECGDataset

# 设置参数
data_dir = 'test_data/'  # 测试数据目录
batch_size = 32  # 批处理大小
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_workers = 4
prefetch_factor = 2  # 可以根据实际情况调整
persistent_workers = True  # 如果你的PyTorch版本支持，可以开启
dataset = ECGDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

# 加载模型
model=torch.load('temp/saved_model/saved.pth', map_location=torch.device('cpu'))

# 定义自定义的 qconfig
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 准备量化
torch.quantization.prepare(model, inplace=True)

# 校准模型
# 用一些校准数据运行模型，以便收集统计数据
model.eval()  # 设置为评估模式
with torch.no_grad():
    for data, _ in tqdm(dataloader, desc="校准模型进度"):
        data = data.to('cpu')  # 确保数据在 CPU 上
        model(data)

model.to('cpu')  # 确保模型在 CPU 上

# 转换量化模型
torch.quantization.convert(model, inplace=True)

model.eval()

# 现在模型已经量化，可以进行推理
# 例如，使用一个新的输入进行推理

# 示例输入数据
input_example = data

# 步骤2：导出PyTorch模型为ExecuTorch格式
aten_dialect = export(model, (input_example,))
edge_program = to_edge(aten_dialect)
executorch_program = edge_program.to_executorch()

# 步骤3：保存导出的模型为.pte文件
output_file = "loongarch/model.pte"
with open(output_file, "wb") as f:
    f.write(executorch_program.buffer)

print(f"ExecuTorch模型已保存为: {output_file}")