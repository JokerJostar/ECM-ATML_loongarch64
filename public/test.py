import os

import torch

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from tqdm.auto import tqdm as tqdmauto
from public.dataset import ECGDataset


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

# 定义评估函数
def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    blue = "\033[94m"
    reset = "\033[0m"
    pbar = tqdmauto(total=len(dataloader), desc="Evaluating", unit="batch",
                    bar_format=f'{blue}{{l_bar}}{{bar:20}}{{r_bar}}{reset}')
    try:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            pbar.update()
    finally:
        pbar.close()  # 无论是否出现错误，都关闭进度条
    return y_true, y_pred


def test_model(model,model_file):
    data_dir = 'ttdata'  # 测试数据目录
    batch_size = 1000  # 批处理大小
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用的设备，cpu或cuda
    output_dir = os.path.join(model_file, 'temp', 'records')  # 输出记录的目录

    num_workers = NUM_WORKERS
    prefetch_factor = PREFETCH_FACTOR  # 可以根据实际情况调整
    persistent_workers = True  # 如果你的PyTorch版本支持，可以开启
    dataset = ECGDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    # 加载模型
    model.to(device)
    model.eval()  # 设置为评估模式

    # 评估模型
    y_true, y_pred = evaluate(model, dataloader, device)
    print('')  # 打印一个空行

    # 计算指标
    f1 = f1_score(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_true, y_pred, average='binary')



    # 转换为0到100之间的整数，并取模100
    max_metric = max(f1, accuracy, precision, recall)
    min_metric = min(f1, accuracy, precision, recall)

    max_metric_transformed = int((max_metric * 100) % 100)
    min_metric_transformed = int((min_metric * 100) % 100)

    # 如果原始值是1.0，并且结果为0，将其设置为100
    if max_metric == 1.0 and max_metric_transformed == 0:
        max_metric_transformed = 100

    if min_metric == 1.0 and min_metric_transformed == 0:
        min_metric_transformed = 100

    # 打印并保存指标
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(output_file, 'w') as f:
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')


    # 调用函数


    print(f'F1 Score: {f1:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    return max_metric_transformed, min_metric_transformed, f1, accuracy, precision, recall
