#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-quantized-script-model.pt>" << std::endl;
        return -1;
    }

    try {
        // 加载量化的TorchScript模型
        torch::jit::script::Module module = torch::jit::load(argv[1]);

        // 确保模型是量化的
        module.eval();

        // 准备输入数据（根据你的模型需要调整）
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::randn({1, 1, 1, 1250}));

        // 运行模型
        at::Tensor output = module.forward(inputs).toTensor();

        // 打印输出
        std::cout << output << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model:\n" << e.what() << std::endl;
        return -1;
    }

    return 0;
}