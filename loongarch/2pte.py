import torch
import torch.nn as nn
import sys
import os
from torch.export import export
from executorch.exir import to_edge

# 获取当前文件所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到系统路径
sys.path.append(parent_dir)

# 现在你可以导入file2.py中的类
from public.model import CNNModel



# 加载模型
model = torch.load("temp/saved_model/saved.pth", map_location=torch.device('cpu'))
model.eval() 

input_example = torch.randn(1,1,1,1250)  # 例子输入

# 步骤2：导出PyTorch模型为ExecuTorch格式
aten_dialect = export(model, (input_example,))
edge_program = to_edge(aten_dialect)
executorch_program = edge_program.to_executorch()

# 步骤3：保存导出的模型为.pte文件
output_file = "loongarch/model.pte"
with open(output_file, "wb") as f:
    f.write(executorch_program.buffer)

print(f"ExecuTorch模型已保存为: {output_file}")

