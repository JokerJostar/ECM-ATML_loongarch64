import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

# 配置模型的字典生成函数
def compute_conv_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1

def generate_config(channel_scaling_factor, fc_scaling_factor, input_size):
    # Define initial input dimensions
    height = input_size
    width = 1  # Since the kernel sizes are (x, 1), width remains 1

    # Compute the output dimensions after each convolutional layer
    height = compute_conv_output_size(height, 6, 2, 0)  # conv1
    height = compute_conv_output_size(height, 5, 2, 0)  # conv2
    height = compute_conv_output_size(height, 4, 2, 0)  # conv3
    height = compute_conv_output_size(height, 4, 2, 0)  # conv4
    height = compute_conv_output_size(height, 4, 2, 0)  # conv5

    # Calculate the number of features for the fully connected layer
    in_features_fc1 = height * width * (8 * channel_scaling_factor)

    return {
        'conv1_out_channels': 1 * channel_scaling_factor,
        'conv2_out_channels': 2 * channel_scaling_factor,
        'conv3_out_channels': 4 * channel_scaling_factor,
        'conv4_out_channels': 8 * channel_scaling_factor,
        'conv5_out_channels': 8 * channel_scaling_factor,
        'fc1_out_features': 5 * fc_scaling_factor,
        'in_features_fc1': in_features_fc1,
    }

class SqueezeExciteAF(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExciteAF, self).__init__()
        reduced_channels = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = F.silu(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return x * scale

class AFNet(nn.Module):
    def __init__(self, config):
        super(AFNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=config['conv1_out_channels'], kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(config['conv1_out_channels'], affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=config['conv1_out_channels'], out_channels=config['conv2_out_channels'], kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(config['conv2_out_channels'], affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=config['conv2_out_channels'], out_channels=config['conv3_out_channels'], kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(config['conv3_out_channels'], affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            SqueezeExciteAF(config['conv3_out_channels'])
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=config['conv3_out_channels'], out_channels=config['conv4_out_channels'], kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(config['conv4_out_channels'], affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=config['conv4_out_channels'], out_channels=config['conv5_out_channels'], kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(config['conv5_out_channels'], affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            SqueezeExciteAF(config['conv5_out_channels'])
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),  # 使用适度的Dropout
            nn.Linear(in_features=config['in_features_fc1'], out_features=config['fc1_out_features']),
            nn.SiLU(inplace=True)  # 使用SiLU激活函数
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=config['fc1_out_features'], out_features=2)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(conv5_output.size(0), -1)

        fc1_output = self.fc1(conv5_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output