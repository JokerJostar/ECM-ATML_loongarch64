# ECM: Electrocardiogram Classification Model

[Chinese](README.md) | English

[Directory Structure](#directory-structure) | [Training Code Structure](#training-code-structure) | [Usage](#usage) | [Installation](#installation) | [Customization](#customization) | [Proprietor](#proprietor) | [Declaration](#declaration)

## Overview
The Electrocardiogram Classification Model (ECM) is a deep learning model designed to classify electrocardiogram (ECG) signals into two categories: normal and atrial fibrillation (AFIB).

AutoTuneML (ATML) is an unsupervised learning framework designed for automatically optimizing hyperparameters. It is designed to simplify and accelerate the development process of machine learning models, helping users achieve optimal model performance without manual intervention by automatically adjusting hyperparameters.

## Directory Structure
The directory structure of this project is as follows:

```
ECM
├── xx-yy                     # Location to store models and score satisfying the standard (xx is the maximum value of the metric, yy is the minimum value)
├── avoid.txt                 # Records hyperparameter configurations that do not meet the standard
├── dataset.py                # Dataset class
├── onnx_log                  # ONNX export logs
├── params.py                 # Hyperparameter settings
├── public                    # Classes and methods
│   ├── model.py              # Model architecture
│   └── test.py               # Definition of basic metric tests
├── __pycache__               # Python cache files
├── README.md                 # Chinese README file
├── README_EN.md              # English README file
├── record_official           # Official metric records
├── temp                      # Temporary folder
│   ├── records               # Basic metric records for each iteration
│   └── saved_model           # Models saved for each iteration
├── test_data                 # Test data
├── test_official.py          # Official metric testing script
├── train_auto.py             # Training script
└── train_data                # Training data
```

## Training Code Structure

1. **Initialization**:
   - Initializing various parameters and settings in the code, such as learning rate, batch size, file paths, and ranges of hyperparameters.

2. **Main Loop**:
   - The main loop continuously samples randomly from the defined ranges to search for the best hyperparameters.
   - It checks specific conditions in the directory structure and gracefully handles interruptions.

3. **Training and Evaluation**:
   - Within the loop, the code trains a CNN model using the selected hyperparameters.
   - After training, it evaluates the performance of the model using validation data.
   - Evaluation metrics such as F1 score, accuracy, precision, and recall are computed.

4. **Folder Renaming**:
   - Based on evaluation metrics, the code checks if the performance meets specific standards.
   - If the performance is below a threshold, it records the hyperparameters in a file to avoid similar configurations in the future.

5. **Training Setup and Loop**:
   - The code sets up data loading, model architecture, loss function, optimizer, and learning rate scheduler.
   - It then trains the model for multiple epochs, updating model parameters according to the optimization algorithm and adjusting learning rates when necessary.

6. **Early Stopping**:
   - The code implements early stopping to prevent overfitting. If the validation loss does not improve over a certain number of epochs, training is stopped early.

7. **ONNX Export**:
   - After training, the code exports the trained model to ONNX format, allowing it to be used in other frameworks or deployed to production environments.

8. **Final Testing**:
   - The code tests the finally trained model on separate test data to ensure its generalization performance.
   - Evaluation metrics are computed and recorded for further analysis.

## Usage
1. **Data Preparation**: Place the training and testing data in the `train_data/` and `test_data/` directories, respectively.
2. **Training**: Run the `train_auto.py` script to start training. During training, model and metric records will be saved in the `temp/` directory.
3. **Testing**: Evaluate the trained model on test data using the `test_official.py` script.
4. **ONNX Export**: After training, the model can be exported to ONNX format. Check the `onnx_log/` directory for export logs.

## Installation
To install the required dependencies, you can create a new conda environment using the following command:

```bash
conda create -n ECM&ATML python=3.10 pytorch==1.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 numpy scikit-learn pandas tqdm onnx -c pytorch -c nvidia
```

## Customization
- **Model Architecture**: Modify the model architecture in the model define (`public/model.py`) according to your requirements.
- **Hyperparameters**: Adjust hyperparameters in the parameter script (`params.py`) according to your requirements.

## Proprietor
- [JokerJostar(Yuqi Lin)](https://github.com/JokerJostar)`2530204503@qq.com`

An undergraduate student from the Class of 2023

## Declaration
Taking the opportunity of participating in the IESD-2024 competition, the entire project was completed independently from scratch by me on May 19, 2024. The model structure, training framework, and hyperparameter optimization algorithm are all original and cannot be used by others without my permission.