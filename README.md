# ECM：心电图分类模型 以及 ATML： 自动优化模型结构和超参数的无监督训练框架 

[English](README_EN.md) | 简体中文

[训练框架功能结构](#训练框架功能结构) | [目录结构](#目录结构) | [使用方法](#使用方法) | [安装](#安装) | [项目所有者](#项目所有者) | [声明](#声明)

## 概述
Electrocardiogram Classification Model（ECM）是一个深度学习模型，旨在将心电图（ECG）信号分类为两类：正常和房颤（AFIB）。

AutoTuneML（ATML）是一个无监督学习框架，可以自动适配cpu和cuda。它旨在简化和加速机器学习模型的开发过程，主要通过以下两点

1. *自动搜索优化模型结构*

2. *自动调整超参数（目前适配了学习率）*

帮助用户在无需手动干预的情况下获得最佳模型性能。并且通过模块化设计保持了灵活性，和易用性。

loongarch部署相关请前往[deploy_loongarch](https://github.com/JokerJostar/deploy_loongarch)









## 训练框架功能结构


1. **Loop Search**:
      - **第一阶段**:
      
      将多种待选的模型结构存入model.py中，通过train_muti.py进行训练，得到各项指标的结果，存入multi-result中。框架会自动尝试不同的学习率，自动迭代模型，自动保存最优模型。此轮只以性能指标为迭代依据，但是会记录onnx文件大小来近似推理速度。
      
      
      - **第二阶段**:
      
      用户从multi-result中根据各项性能指标和文件大小来挑选合适的模型结构，导入ss_model.py中，作为此轮的模型基础结构。接下来设定预期的性能指标和文件大小，通过train_final.py进行训练，框架在指定的范围内自动调整卷积层和全连接层的通道数，直到模型性能与文件大小同时达标。此轮以性能指标和文件大小为迭代依据。

      **优势**：

      1. 给出多种基础模型后，框架可以通过Loop Search的第一阶帮助用户找到最适合本次任务的模型结构。

      2. 在Loop Search第二阶段，框架可以通过调整模型复杂程度，自动帮用户找到性能指标与推理速度的平衡点。

      3. 节省了大量结构调优的时间，模型进化速度非常快。

2. **early stop**:
   
   为了防止过拟合，框架采用了early stop策略。但是传统的early stop逻辑只监控了val_loss,并不能有效地防止过拟合。因此框架采用了一种新的early stop逻辑，拥有更好的性能。

3. **超参数裁枝**:

   为了节省计算资源与时间，框架采用了超参数裁枝策略。框架会对效果不佳的超参数及其邻域进行裁剪，以提高训练效率。



## 目录结构
该项目的目录结构如下：

```
├── 93-89                   #Loop Search第二阶段的结果（xx-yy,其中xx为各项指标的最大值，yy为最小值）
├── avoid.txt               #超参数裁枝定义
├── multi-result            #loop search第一阶段的结果
├── params.py               #各参数的设置
├── public                  #类和定义
│     ├── dataset.py        #数据集定义
│     ├── model.py          #loop search第一阶段模型定义
│     ├── ss_model.py       #loop search第二阶段模型定义 
│     └── test.py           #测试函数定义       
├── temp                    #Loop Search第二阶段的临时文件夹
├── test_data               #正式测试集
├── train_data              #正式训练集
├── train_final.py          #loop search第二阶段
├── train_muti.py           #loop search第一阶段
└── ttdata                  #演示数据集

```


## 使用方法
1. **数据准备**：将训练和测试数据分别放置在`train_data/`和`test_data/`目录中。（默认是演示数据集`ttdata`）

2. **参数设置**：在`params.py`中设置超参数。

3. **模型定义**：在`public/model.py`中定义模型结构,并将模型注册进代码末尾的`model_classes`中。

3. **训练**：运行`train_multi.py`脚本以开始训练。训练过程中，模型和指标记录将保存在`multi-result`目录中。挑选合适的模型结构并将其导入`ss_model.py`中。设定期望的性能指标和文件大小，然后运行`train_final.py`脚本以开始第二阶段的训练。结果将保存在`xx-yy`目录中。

## 安装
要安装所需的依赖项，您可以使用以下命令创建一个新的conda环境：

```bash
conda create -n ECM&ATML python=3.10 pytorch==1.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 numpy scikit-learn pandas tqdm onnx -c pytorch -c nvidia
```

## 项目所有者
- [JokerJostar(林宇祺)](https://github.com/JokerJostar)`2530204503@qq.com`2023级大一

## 声明
整个项目以参加IESD-2024大赛为契机，由本人自2024年5月19日从零开始独立完成，欢迎交流。
