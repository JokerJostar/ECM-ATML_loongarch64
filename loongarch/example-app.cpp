// 引入Torch头文件，Tensor类在此头文件中，别的类会在另外的头文件中
#include <torch/torch.h>
#include <iostream>

int main() {
  // 使用arange构造一个一维向量，再用reshape变换到5x5的矩阵
  torch::Tensor foo = torch::arange(25).reshape({5, 5});

  // 计算矩阵的迹
  torch::Tensor bar  = torch::einsum("ii", foo);

  // 输出矩阵和对应的迹
  std::cout << "==> matrix is:\n " << foo << std::endl;
  std::cout << "==> trace of it is:\n " << bar << std::endl;
}