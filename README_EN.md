# ECM: Electrocardiogram Classification Model and ATML: Automated Hyperparameter Optimization for Unsupervised Training Framework

[简体中文](README.md) | English

[Training Framework Functional Structure](#training-framework-functional-structure) | [Directory Structure](#directory-structure) | [Usage](#usage) | [Installation](#installation) | [Project Owner](#project-owner) | [Disclaimer](#disclaimer)

## Overview
The Electrocardiogram Classification Model (ECM) is a deep learning model designed to classify ECG signals into two categories: Normal and Atrial Fibrillation (AFIB).

AutoTuneML (ATML) is an unsupervised learning framework that automatically adapts to both CPU and CUDA. It aims to simplify and accelerate the development of machine learning models, primarily through the following two features:

1. *Automatically search and optimize model structures*

2. *Automatically adjust hyperparameters (currently supports learning rate)*

This helps users achieve optimal model performance without manual intervention, while maintaining flexibility and ease of use through modular design.

For loongarch deployment, please visit [deploy_loongarch](https://github.com/JokerJostar/deploy_loongarch).

## Training Framework Functional Structure

1. **Loop Search**:
      - **First Phase**:
      
      Store various candidate model structures in `model.py` and train them using `train_muti.py`, storing the results in `multi-result`. The framework will automatically try different learning rates, iterate through the models, and save the optimal model. This phase uses performance metrics as the iteration criterion but also records the size of the ONNX files to approximate inference speed.
      
      
      - **Second Phase**:
      
      Users select the appropriate model structure from `multi-result` based on performance metrics and file size, and import it into `ss_model.py` as the base model structure for this phase. Then, set the expected performance metrics and file size, and train using `train_final.py`. The framework will automatically adjust the number of channels in convolutional and fully connected layers within the specified range until both performance and file size meet the criteria. This phase uses both performance metrics and file size as iteration criteria.

      **Advantages**:

      1. Given multiple base models, the framework can help users find the most suitable model structure for the task through the first phase of Loop Search.

      2. In the second phase of Loop Search, the framework can automatically find the balance between performance metrics and inference speed by adjusting the model complexity.

      3. Saves a significant amount of time on structural tuning, with a very fast model evolution speed.

2. **Early Stop**:
   
   To prevent overfitting, the framework adopts an early stop strategy. Unlike traditional early stop logic that only monitors `val_loss`, this framework uses a new early stop logic with better performance.

3. **Hyperparameter Pruning**:

   To save computational resources and time, the framework employs a hyperparameter pruning strategy. It prunes ineffective hyperparameters and their neighborhoods to improve training efficiency.

## Directory Structure
The project's directory structure is as follows:

```
├── 100-46                  # Results of the second phase of Loop Search (xx-yy, where xx is the maximum value of various metrics, and yy is the minimum value)
├── avoid.txt               # Definition of hyperparameter pruning
├── multi-result            # Results of the first phase of Loop Search
├── params.py               # Parameter settings
├── public                  # Classes and definitions
│     ├── dataset.py        # Dataset definition
│     ├── model.py          # Model definition for the first phase of Loop Search
│     ├── ss_model.py       # Model definition for the second phase of Loop Search
│     └── test.py           # Test function definition
├── temp                    # Temporary files for the second phase of Loop Search
├── test_data               # Official test set
├── train_data              # Official training set
├── train_final.py          # Second phase of Loop Search
├── train_muti.py           # First phase of Loop Search
└── ttdata                  # Demonstration dataset

```

## Usage
1. **Data Preparation**: Place the training and test data in the `train_data/` and `test_data/` directories, respectively. (The default is the demonstration dataset `ttdata`)

2. **Parameter Settings**: Set hyperparameters in `params.py`.

3. **Model Definition**: Define the model structure in `public/model.py` and register the models at the end of the file in `model_classes`.

3. **Training**: Run the `train_multi.py` script to start training. During training, the models and metric records will be saved in the `multi-result` directory. Select an appropriate model structure and import it into `ss_model.py`. Set the desired performance metrics and file size, and then run the `train_final.py` script to start the second phase of training. The results will be saved in the `xx-yy` directory.

## Installation
To install the required dependencies, you can create a new conda environment with the following command:

```bash
conda create -n ECM&ATML python=3.10 pytorch==1.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 numpy scikit-learn pandas tqdm onnx -c pytorch -c nvidia
```

## Project Owner
- [JokerJostar (Lin Yuqi)](https://github.com/JokerJostar) `2530204503@qq.com` Class of 2023 Freshman

## Disclaimer
This project was independently developed by me from scratch starting on May 19, 2024, for the IESD-2024 competition. I welcome any discussions and exchanges.