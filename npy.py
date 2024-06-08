import numpy as np
import os
import onnx

# 加载 ONNX 模型
model = onnx.load('adjusted_model.onnx')

# 打印模型的输入信息
for input in model.graph.input:
    print(input.name)
    print(input.type.tensor_type.shape.dim)


# 指定包含TXT文件的目录路径
directory_path = 'train_data'

# 获取目录中的所有文件，并按顺序加载前100个文件
file_paths = sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path)])[:100]

# 检查是否确实找到了100个文件
assert len(file_paths) == 100, f"Expected 100 files, but found {len(file_paths)} files."

# 初始化一个空列表来存储所有文件的数据
data_list = []

for file_path in file_paths:
    # 读取每个文件的1250个数据点
    data = np.loadtxt(file_path)
    # 确保数据是 (1250,) 形状
    assert data.shape == (1250,), f"Unexpected shape in {file_path}: {data.shape}"
    # 添加到数据列表
    data_list.append(data)

# 将数据列表转换为NumPy数组并重新形状为 (100, 1, 1250, 1)
data_array = np.array(data_list).reshape(100, 1, 1250, 1)

# 计算均值和标准差
mean = np.mean(data_array)
std = np.std(data_array)

# 打印均值和标准差
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")

# 保存为npy文件
npy_file_path = 'data.npy'
np.save(npy_file_path, data_array)

print(f'Data saved to {npy_file_path} with shape {data_array.shape}')
