import os
import re
import sys
import signal
import shutil
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm as tqdmauto
from itertools import product

from public.ss_model import  AFNet as Net
from public.ss_model import generate_config
from public.dataset import ECGDataset
from public.test import test_model
import math

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
    AVOID_PARAM,
    LEARING_RATE
)



# Signal handler for graceful exit
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)








# Check and rename folders based on metrics
def check_and_rename_folder(max_metric, min_metric,size, root_dir='.'):

    folder_found = False

    for folder_name in os.listdir(root_dir):
        match = re.match(r'(\d+)-(\d+)', folder_name)
        if match:
            folder_found = True
            xx, yy = map(int, match.groups())
            if  min_metric >= yy and size <= 60:
                new_folder_name = f'{int(max_metric)}-{int(min_metric)}'
                os.rename(os.path.join(root_dir, folder_name), os.path.join(root_dir, new_folder_name))
                for file_name in os.listdir(os.path.join(root_dir, new_folder_name)):
                    if file_name.endswith('.pth'):
                        shutil.copy(f'temp/saved_model/{file_name}', os.path.join(root_dir, new_folder_name, file_name))
                    elif file_name.endswith('.onnx'):
                        shutil.copy(f'temp/saved_model/{file_name}', os.path.join(root_dir, new_folder_name, file_name))
                    elif file_name.endswith('.txt'):
                        shutil.copy(f'temp/records/{file_name}', os.path.join(root_dir, new_folder_name, file_name))


    if not folder_found:
        new_folder_name = f'{int(max_metric)}-{int(min_metric)}'
        os.makedirs(os.path.join(root_dir, new_folder_name))
        for file_name in os.listdir(os.path.join(root_dir, 'temp/saved_model')):
            if file_name.endswith('.pth') or file_name.endswith('.onnx'):
                shutil.copy(os.path.join(root_dir, 'temp/saved_model', file_name), os.path.join(root_dir, new_folder_name, file_name))
        for file_name in os.listdir(os.path.join(root_dir, 'temp/records')):
            if file_name.endswith('.txt'):
                shutil.copy(os.path.join(root_dir, 'temp/records', file_name), os.path.join(root_dir, new_folder_name, file_name))


def save_model_as_onnx(model, model_save_path, dataloader):
    # 确保模型处于评估模式
    model.eval()
    
    # 从 DataLoader 中获取一个批次作为示例输入
    inputs, _ = next(iter(dataloader))
    dummy_input = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 导出模型为 ONNX 格式
    onnx_save_path = model_save_path + '.onnx'
    torch.onnx.export(model, dummy_input, onnx_save_path, opset_version=11, input_names=['input'], output_names=['output'])
    print(f'Saved model in ONNX format at {onnx_save_path}')

# Main training loop
def main():
    patience_counter = 0

    # 生成所有可能的 channel_scaling_factor 和 fc_scaling_factor 组合
    unique_combinations = list(product(range(1, 6), repeat=2))
    PATIENCE_COUNTER_MAX = len(unique_combinations)

    while patience_counter < PATIENCE_COUNTER_MAX:
        for channel_scaling_factor, fc_scaling_factor in unique_combinations:
            print(f"Running model with channel_scaling_factor={channel_scaling_factor}, fc_scaling_factor={fc_scaling_factor}")
            
            config = generate_config(channel_scaling_factor, fc_scaling_factor,1250)

            learning_rate = LEARING_RATE

            try:
                for folder_name in os.listdir('.'):
                    match = re.match(r'(\d+)-(\d+)', folder_name)
                    if match:
                        xx, yy = map(int, match.groups())
                        if yy >= 100:
                            print('yy is satisfied, exiting...')
                            sys.exit(0)
                        else:
                            patience_counter += 1
                            if patience_counter > PATIENCE_COUNTER_MAX:
                                print('All combinations processed, exiting...')
                                sys.exit(0)

                print(f'patience_counter: {patience_counter}/{PATIENCE_COUNTER_MAX}')

                dataset = ECGDataset(DATA_DIR)
                indices = list(range(len(dataset)))
                train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
                train_subset = Subset(dataset, train_indices)
                val_subset = Subset(dataset, val_indices)

                train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                          prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)
                onnx_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS,
                                          prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)
                val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Net(config).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

                best_val_loss = float('inf')
                patience = 3
                no_improve_epochs = 0
                total_steps = NUM_EPOCHS * len(train_loader)

                with tqdmauto(total=total_steps, desc="Training", unit="step") as pbar:
                    for epoch in range(NUM_EPOCHS):
                        model.train()
                        running_loss = 0.0
                        for inputs, labels in train_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                            pbar.update(1)
                        avg_train_loss = running_loss / len(train_loader)

                        val_loss = 0.0
                        model.eval()
                        with torch.no_grad():
                            for inputs, labels in val_loader:
                                inputs, labels = inputs.to(device), labels.to(device)
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                                val_loss += loss.item()
                        avg_val_loss = val_loss / len(val_loader)

                        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, patience_counter: {patience_counter}/{PATIENCE_COUNTER_MAX}')
                        scheduler.step(avg_val_loss)

                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            no_improve_epochs = 0
                        else:
                            no_improve_epochs += 1
                            if no_improve_epochs >= patience and avg_train_loss < avg_val_loss - 0.01:
                                print('Early stopping\n')
                                break

                if not os.path.exists('temp/saved_model'):
                    os.makedirs('temp/saved_model')
                torch.save(model, MODEL_SAVE_PATH + '.pth')
                save_model_as_onnx(model, MODEL_SAVE_PATH, onnx_loader)
                
                print('Saved model in .pth format at the end of training')
                max_metric_transformed, min_metric_transformed, f1, accuracy, precision, recall = test_model(model,'.')

                output_dir = 'temp/records/'
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, 'evaluation_metrics.txt')
                onnx_file_path = os.path.join(MODEL_SAVE_PATH + '.onnx')
                onnx_file_size = os.path.getsize(onnx_file_path) / 1024  # Convert bytes to kilobytes

                with open(output_file, 'a') as f:
                    f.write(f'ONNX file size: {onnx_file_size:.2f} KB\n')

                
                check_and_rename_folder(max_metric_transformed, min_metric_transformed, onnx_file_size)

                patience_counter += 1

            except KeyboardInterrupt:
                print('KeyboardInterrupt detected, exiting...')
                sys.exit(0)

    print('All combinations processed, exiting...')
    sys.exit(0)

if __name__ == '__main__':
    main()