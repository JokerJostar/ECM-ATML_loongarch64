# ECM：心电图分类模型 以及 ATML： 自动优化超参数的无监督训练框架 

[English](README_EN.md) | 简体中文

[目录结构](#目录结构) | [模型选择](#模型选择) | [训练代码结构](#训练代码结构) | [使用方法](#使用方法) | [安装](#安装) | [定制](#定制) | [项目所有者](#项目所有者) | [声明](#声明)

## 概述
Electrocardiogram Classification Model（ECM）是一个深度学习模型，旨在将心电图（ECG）信号分类为两类：正常和房颤（AFIB）。

AutoTuneML（ATML）是一个无监督学习框架，专为自动优化超参数而设计，可以自动适配cpu和cuda。它旨在简化和加速机器学习模型的开发过程，通过自动调整超参数（目前适配了学习率），帮助用户在无需手动干预的情况下获得最佳模型性能。并且通过模块化设计保持了灵活性，和易用性。

loongarch部署相关请前往[deploy_loongarch](https://github.com/JokerJostar/deploy_loongarch)

## 目录结构
该项目的目录结构如下：

```
ECM
├── xx-yy                     # 存储满足标准的模型和指标分数的位置（xx为指标的最大值，yy为最小值）
├── avoid.txt                 # 记录不满足标准的超参数配置
├── loongarch                 # loongarch适配转换
├── public                    # 类和方法
│   ├── dataset.py            # 数据集类
│   ├── model.py              # 模型架构
│   └── test.py               # 基本指标测试定义
├── __pycache__               # Python缓存文件
├── README.md                 # 中文说明文件
├── README_EN.md              # 英文说明文件
├── record_official           # 官方指标记录
├── temp                      # 临时文件夹
│   ├── records               # 每个循环的基本指标记录
│   └── saved_model           # 每个循环保存的模型
├── test_data                 # 测试数据
├── test_official.py          # 官方指标测试脚本
├── train_auto.py             # 训练脚本
└── train_data                # 训练数据
```



## 模型选择

### blocknet

ShuffleNet
本针对龙芯平台设计了两个版本的net

1. 保留了channel shuffle和block结构的shufflenet

2. 去除了channel shuffle但保留了block结构的blocknet

结果是channel shuffle在低算力平台对精度几乎没有影响

这个结构的模型量化后大小为9kb，训练轮数少，3个epoch就可以收敛到位，精度在0.9左右，但是量化对其精度影响大。

### 单程线性的卷积网络

简单的卷积网络，体积稍大但是结构简单，速度更快

### 其他

见 `public/model.py`







## 训练代码结构


1. **初始化**：
   - 代码初始化各种参数和设置，如学习率、批处理大小、文件路径和超参数范围等。

2. **主循环**：
   - 主循环持续不断地从定义的范围中随机抽样，以寻找最佳超参数。
   - 它检查目录结构中的特定条件，并优雅地处理中断。

3. **训练和评估**：
   - 在循环内部，代码使用选定的超参数训练CNN模型。
   - 训练后，它使用验证数据评估模型的性能。
   - 计算评估指标，如F1分数、准确率、精确度和召回率。

4. **文件夹重命名**：
   - 基于评估指标，代码检查性能是否达到特定标准。
   - 如果性能低于阈值，则将超参数记录在文件中，以避免将来类似的超参数配置。

5. **训练设置和循环**：
   - 代码设置数据加载、模型架构、损失函数、优化器和学习率调度器。
   - 然后，它对模型进行多个时期的训练，根据优化算法更新模型参数，并在生成学习率时加入裁枝算法，节省训练时间。

6. **提前停止**：
   - 代码实现了提前停止以防止过拟合。如果验证损失在一定数量的时期内没有改善，则提前停止训练。

7. **最终测试**：
   - 代码在单独的测试数据上测试最终训练好的模型，以确保其泛化性能。
   - 计算评估指标，并记录以供进一步分析。

## 使用方法
1. **数据准备**：将训练和测试数据分别放置在`train_data/`和`test_data/`目录中。
2. **训练**：运行`train_auto.py`脚本以开始训练。训练过程中，模型和指标记录将保存在`temp/`目录中。
3. **测试**：使用`test_official.py`脚本在测试数据上评估训练好的模型。

## 安装
要安装所需的依赖项，您可以使用以下命令创建一个新的conda环境：

```bash
conda create -n ECM&ATML python=3.10 pytorch==1.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 numpy scikit-learn pandas tqdm onnx -c pytorch -c nvidia
```

## 定制
- **模型架构**：根据您的要求在模型定义（`public/model.py`)中修改模型架构。
- **超参数**：根据您的要求在参数脚本（`params.py`)中修改超参数。

## 项目所有者
- [JokerJostar(林宇祺)](https://github.com/JokerJostar)`2530204503@qq.com`2023级大一

## 声明
整个项目以参加IESD-2024大赛为契机，由本人自2024年5月19日从零开始独立完成，模型结构，训练框架与超参优化算法等均为原创，他人未经本人许可不得使用。
