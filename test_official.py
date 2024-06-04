import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # 进度条库
from public.dataset import ECGDataset  # 确保你有这个文件
from public.model import CustomShuffleNetV2 as Net
import torch.quantization
from params import (
    AVOID_FILE_PATH,
    DATA_DIR,
    MODEL_SAVE_PATH,
    INPUT_SIZE,
    NUM_EPOCHS,
    BATCH_SIZE,
    LR_MIN,
    LR_MAX,
    STEP,
    NUM_WORKERS,
    PREFETCH_FACTOR,
)

# 评估指标函数
def ACC(mylist):
    tp, fn, fp, tn = mylist
    total = sum(mylist)
    return (tp + tn) / total

def PPV(mylist):
    tp, fn, fp, tn = mylist
    if tp + fn == 0:
        return 1
    elif tp + fp == 0 and tp + fn != 0:
        return 0
    return tp / (tp + fp)

def NPV(mylist):
    tp, fn, fp, tn = mylist
    if tn + fp == 0:
        return 1
    elif tn + fn == 0 and tn + fp != 0:
        return 0
    return tn / (tn + fn)

def Sensitivity(mylist):
    tp, fn, fp, tn = mylist
    if tp + fn == 0:
        return 1
    return tp / (tp + fn)

def Specificity(mylist):
    tp, fn, fp, tn = mylist
    if tn + fp == 0:
        return 1
    return tn / (tn + fp)

def BAC(mylist):
    return (Sensitivity(mylist) + Specificity(mylist)) / 2

def F1(mylist):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        return 0
    return (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall)

def stats_report(mylist):
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'

    print(output)
    # 创建新的目录
    output_dir = './record_official'
    os.makedirs(output_dir, exist_ok=True)

    # 打开新的文件并写入指标
    output_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(output_file, 'w') as f:
        f.write(output)

    return output


if __name__ == '__main__':
    data_dir = './test_data/'  # 测试数据目录
    batch_size = BATCH_SIZE  # 批处理大小
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_workers = NUM_WORKERS
    prefetch_factor = PREFETCH_FACTOR  # 可以根据实际情况调整
    persistent_workers = True  # 如果你的PyTorch版本支持，可以开启
    dataset = ECGDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    model = Net()
    model.load_state_dict(torch.load('loongarch/quantized_model.pth'))
    model = model.to(device)  # Move the model to the same device as the input data

    model.eval()  # 设置为评估模式

    # 评估模型
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Evaluating"):  # 加入进度条
            data = data.to(device)  # Ensure the input data is on the same device as the model
            outputs = model(data)  # Now you can pass the data to the model
            print("Output Tensor:", outputs)
            print("Output Shape:", outputs.shape)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())  # 确保从GPU传输到CPU

    # 计算混淆矩阵
    tp = sum((pred == 1 and label == 1) for pred, label in zip(all_preds, all_labels))
    fn = sum((pred == 0 and label == 1) for pred, label in zip(all_preds, all_labels))
    fp = sum((pred == 1 and label == 0) for pred, label in zip(all_preds, all_labels))
    tn = sum((pred == 0 and label == 0) for pred, label in zip(all_preds, all_labels))

    # 打印统计报告
    mylist = [tp, fn, fp, tn]
    report = stats_report(mylist)
