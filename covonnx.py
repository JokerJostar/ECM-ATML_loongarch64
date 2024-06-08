import onnx
from onnx import shape_inference

# 加载 ONNX 模型
onnx_model_path = '99-75/saved.onnx'
onnx_model = onnx.load(onnx_model_path)

# 获取输入节点
input_name = None
for input in onnx_model.graph.input:
    input_name = input.name
    break

# 设置新的输入形状（从 NCHW 调整为 NHWC）
new_input_shape = [100, 1, 1, 1250]  # NHWC

# 更新输入节点的形状
for input in onnx_model.graph.input:
    if input.name == input_name:
        input.type.tensor_type.shape.dim[0].dim_value = new_input_shape[0]
        input.type.tensor_type.shape.dim[1].dim_value = new_input_shape[1]
        input.type.tensor_type.shape.dim[2].dim_value = new_input_shape[2]
        input.type.tensor_type.shape.dim[3].dim_value = new_input_shape[3]
        break

# 运行形状推断
inferred_model = shape_inference.infer_shapes(onnx_model)

# 保存调整后的 ONNX 模型
onnx.save(inferred_model, 'adjusted_model.onnx')
