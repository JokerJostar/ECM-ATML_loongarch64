import tensorflow as tf
import numpy as np

# 加载 SavedModel
saved_model_dir = "tflite/"
loaded_model = tf.saved_model.load(saved_model_dir)



# 定义代表性数据生成器函数
def representative_data_gen():
    for _ in range(200):  # 生成50个样本
        # 生成随机数据，这里假设输入是你模型的输入形状
        input_data = np.random.rand(batch_size, input_height, input_width, input_channels).astype(np.float32)
        yield [input_data]

# 定义输入数据的形状
batch_size = 100  # 批大小
input_height = 1  # 输入高度
input_width = 1250  # 输入宽度
input_channels = 1  # 输入通道数

# 转换为 TFLite 模型
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# 设置代表性数据集
converter.representative_dataset = representative_data_gen

# 转换模型
tflite_quantized_model = converter.convert()

# 保存量化后的模型
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_quantized_model)
