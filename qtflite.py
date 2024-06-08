import tensorflow as tf
import numpy as np
import random
import os

# 加载 SavedModel
saved_model_dir = "tflite/"
loaded_model = tf.saved_model.load(saved_model_dir)



# 定义代表性数据生成器函数
def representative_data_gen():
    train_data_folder = 'train_data'
    txt_files = [f for f in os.listdir(train_data_folder) if f.endswith('.txt')]
    
    # 随机选择80个文件
    selected_files = random.sample(txt_files, 80)
    
    for file_name in selected_files:
        file_path = os.path.join(train_data_folder, file_name)
        
        # 读取文件中的浮点数
        with open(file_path, 'r') as file:
            data = file.readlines()
        
        # 将数据转换为numpy数组，并改变形状
        data = np.array([float(x.strip()) for x in data], dtype=np.float32)
        input_data = data.reshape((1, 1250, 1, 1))  # batch_size=1, 高=1250, 宽=1, 通道=1
        
        yield [input_data]

# 定义输入数据的形状
batch_size = 1  # 批大小
input_height = 1250  # 输入高度
input_width = 1  # 输入宽度
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
