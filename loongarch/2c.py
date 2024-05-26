import os

def convert_model_to_cpp_array(model_path, output_path):
    with open(model_path, 'rb') as f:
        model_data = f.read()

    array_str = ', '.join(f'0x{byte:02x}' for byte in model_data)
    array_len = len(model_data)

    cpp_content = f"""
    #ifndef MODEL_DATA_H
    #define MODEL_DATA_H

    #include <cstddef>

    const unsigned char model_data[{array_len}] = {{
        {array_str}
    }};
    const size_t model_data_len = {array_len};

    #endif  // MODEL_DATA_H
    """

    with open(output_path, 'w') as f:
        f.write(cpp_content)

if __name__ == "__main__":
    model_path = "D:/ECM_and_ATML/loongarch/saved_net.pt"  # 替换为你的模型文件路径
    output_path = "D:/ECM_and_ATML/loongarch/model_data.h"
    convert_model_to_cpp_array(model_path, output_path)
