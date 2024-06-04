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

from public.model import CNNModel
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
)

# The maximum number of times the patience counter can increment before stopping training
PATIENCE_COUNTER_MAX = math.ceil((LR_MAX-LR_MIN)/STEP)

# Signal handler for graceful exit
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# File operations for reading and writing avoid values
def read_avoid_values(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file.readlines()]

def write_avoid_value(file_path, value):
    with open(file_path, 'a') as file:
        file.write(f'{value}\n')

# Validate learning rate against avoid values
def is_valid_learning_rate(learning_rate, avoid_values):
    for value in avoid_values:
        if value - (2*STEP) <= learning_rate <= value + (2*STEP):
            return False
    return True

# Generator for learning rate values
def learning_rate_generator(lr_min, lr_max, step, avoid_values):
    lr = float(lr_min)
    while lr <= lr_max:
        if is_valid_learning_rate(lr, avoid_values):
            yield lr
        lr += float(step)

# Check and rename folders based on metrics
def check_and_rename_folder(max_metric, min_metric, root_dir='.'):
    for folder_name in os.listdir(root_dir):
        match = re.match(r'(\d+)-(\d+)', folder_name)
        if match:
            xx, yy = map(int, match.groups())
            if 0 <= xx <= 100 and 0 <= yy <= 100 and min_metric > yy:
                new_folder_name = f'{int(max_metric)}-{int(min_metric)}'
                os.rename(os.path.join(root_dir, folder_name), os.path.join(root_dir, new_folder_name))
                for file_name in os.listdir(os.path.join(root_dir, new_folder_name)):
                    if file_name.endswith('.pth'):
                        shutil.copy(f'temp/saved_model/{file_name}', os.path.join(root_dir, new_folder_name, file_name))
                    elif file_name.endswith('.txt'):
                        shutil.copy(f'temp/records/{file_name}', os.path.join(root_dir, new_folder_name, file_name))

# Main training loop
def main():
    patience_counter = 0

    avoid_values = read_avoid_values(AVOID_FILE_PATH)
    lr_gen = learning_rate_generator(LR_MIN, LR_MAX, STEP, avoid_values)

    while True:
        learning_rate = next(lr_gen, None)
        if learning_rate > LR_MAX:
            print("Training finished")
            break

        try:
            for folder_name in os.listdir('.'):
                match = re.match(r'(\d+)-(\d+)', folder_name)
                if match:
                    xx, yy = map(int, match.groups())
                    if yy >= 97:
                        print('yy is satisfied, exiting...')
                        sys.exit(0)
                    else:
                        patience_counter += 1
                        if patience_counter >= PATIENCE_COUNTER_MAX:
                            print('training finished, exiting...')
                            sys.exit(0)

            print(f'\n\nLearning Rate: {learning_rate}')
            print(f'Batch Size: {BATCH_SIZE}')
            print(f'patience_counter: {patience_counter}/{PATIENCE_COUNTER_MAX}')

            dataset = ECGDataset(DATA_DIR)
            indices = list(range(50000))
            train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)

            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                      prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                    prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = CNNModel(INPUT_SIZE).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

            best_val_loss = float('inf')
            patience = 5
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
                        if no_improve_epochs >= patience:
                            print('Early stopping\n')
                            break

            if not os.path.exists('temp/saved_model'):
                os.makedirs('temp/saved_model')
            torch.save(model, MODEL_SAVE_PATH + '.pth')
            
            print('Saved model in .pth format at the end of training')

            output_dir = './temp/records/'
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'evaluation_metrics.txt')

            with open(output_file, 'a') as f:
                f.write(f'Learning rate: {learning_rate:.4f}\n')

            max_metric_transformed, min_metric_transformed, f1, accuracy, precision, recall = test_model(model)
            check_and_rename_folder(max_metric_transformed, min_metric_transformed)

            if min(f1, accuracy, precision, recall) < AVOID_PARAM:
                write_avoid_value(AVOID_FILE_PATH, learning_rate)

        except KeyboardInterrupt:
            print('KeyboardInterrupt detected, exiting...')
            sys.exit(0)

if __name__ == '__main__':
    main()