import torch
import torch.nn as nn
import sys
import os
from torch.export import export
from executorch.exir import to_edge
from torch.utils.data import Dataset, DataLoader

# 获取当前文件所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到系统路径
sys.path.append(parent_dir)

# 现在你可以导入file2.py中的类
from public.dataset import ECGDataset
from public.model import ShuffleNetV2 as Net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载模型
modelf = Net()  # 确保你已经定义了 Net 类
modelf.qconfig = torch.quantization.get_default_qconfig('fbgemm')
modelp = torch.quantization.prepare(modelf)
model = torch.quantization.convert(modelp)
model.load_state_dict(torch.load('loongarch/quantized_model.pth'))
model.to(device)  # 确保模型在 GPU 上

model.eval()  # 设置为评估模式

# 设置数据加载参数
data_dir = './test_data/'  # 测试数据目录
batch_size = 100  # 批处理大小


num_workers = 4
prefetch_factor = 2  # 可以根据实际情况调整
persistent_workers = True  # 如果你的PyTorch版本支持，可以开启

# 加载数据集和数据加载器
dataset = ECGDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, drop_last=True)

# 使用 DataLoader 中的一个批次数据
for data, labels in dataloader:
    data = data.to(device)
    break  # 只需要一个批次数据


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

