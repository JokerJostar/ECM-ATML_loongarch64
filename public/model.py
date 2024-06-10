import torch
import torch.nn as nn
import torchvision.models as models
import torch.ao.quantization
from torchvision.models import ShuffleNet_V2_X1_0_Weights
import torch.quantization
import torch.nn.functional as F
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExciteAF(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExciteAF, self).__init__()
        reduced_channels = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return x * scale

class AFNet(nn.Module):
    def __init__(self):
        super(AFNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(4, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(8, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            SqueezeExciteAF(16)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            SqueezeExciteAF(32)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),  # 使用适度的Dropout
            nn.Linear(in_features=1184, out_features=10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(conv5_output.size(0), -1)

        fc1_output = F.leaky_relu(self.fc1(conv5_output), negative_slope=0.1)
        fc2_output = self.fc2(fc1_output)
        return fc2_output
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcite, self).__init__()
        reduced_channels = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return x * scale

class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleNetV2Block, self).__init__()
        self.stride = stride
        mid_channels = out_channels // 2

        if self.stride == 1:
            self.branch_main = nn.Sequential(
                DepthwiseSeparableConv(in_channels // 2, mid_channels),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch_main = nn.Sequential(
                DepthwiseSeparableConv(in_channels, mid_channels, stride=stride),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.branch_proj = nn.Sequential(
                DepthwiseSeparableConv(in_channels, in_channels, stride=stride),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat((x1, self.branch_main(x2)), dim=1)
        else:
            out = torch.cat((self.branch_proj(x), self.branch_main(x)), dim=1)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(ShuffleNetV2, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.stage2 = self._make_stage(16, 32, 2)  # 增加块数
        self.stage3 = self._make_stage(32, 64, 2)  # 增加块数
        self.stage4 = self._make_stage(64, 128, 1)  # 保持单个块

        self.se = SqueezeExcite(128)
        self.fc = nn.Linear(128, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(ShuffleNetV2Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.se(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.fc(x)
        return x

class OptimizedAFNet(nn.Module):
    def __init__(self):
        super(OptimizedAFNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(7, 1), stride=(2, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm2d(8, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 1), stride=(2, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=16),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=16, out_features=2)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)

        pooled_output = self.global_avg_pool(conv3_output)
        pooled_output = pooled_output.view(pooled_output.size(0), -1)  # Flatten the tensor

        fc1_output = self.fc1(pooled_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=1, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(32 * 312 * 5, 128)  # 根据打印的形状调整输入大小
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 形状变为 (batch_size, 16, 625, 1)
        x = self.pool(F.relu(self.conv2(x)))  # 形状变为 (batch_size, 32, 312, 5)


        x = x.view(x.size(0), -1)  # 自动计算展平后的大小
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x




class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleResNet, self).__init__()
        self.in_channels = 32  # Reduced initial number of channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 1)  # Reduced number of blocks
        self.layer2 = self._make_layer(64, 1, stride=2)  # Reduced number of blocks
        self.layer3 = self._make_layer(128, 1, stride=2)  # Reduced number of blocks
        self.layer4 = self._make_layer(256, 1, stride=2)  # Reduced number of blocks

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)  # Reduced the number of neurons

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # Define the layers according to the diagram
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=(2, 2))  # Input: 1x1250x1x1, Output: 1x625x1x4
        self.conv2a = nn.Conv2d(4, 4, kernel_size=(3, 3), padding=(2, 2))  # Input: 1x627x3x4, Output: 1x313x1x4
        self.conv2b = nn.Conv2d(4, 4, kernel_size=(3, 3), padding=(2, 2))  # Input: 1x627x3x4, Output: 1x313x1x4
        self.conv3a = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=(2, 2))  # Input: 1x315x3x8, Output: 1x157x1x8
        self.conv3b = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=(2, 2))  # Input: 1x315x3x8, Output: 1x157x1x8
        
        self.fc = nn.Linear(16, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.pad(x, (2, 2, 2, 2))
        
        x1 = F.relu(self.conv2a(x))
        x2 = F.relu(self.conv2b(x))
        
        x = torch.cat((x1, x2), dim=1)
        x = F.pad(x, (2, 2, 2, 2))
        
        x1 = F.relu(self.conv3a(x))
        x2 = F.relu(self.conv3b(x))
        
        x = torch.cat((x1, x2), dim=1)
        
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class SimplifiedMobileNetBinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SimplifiedMobileNetBinaryClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(16, 32, stride=1),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio
        
        self.use_residual = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(Swish())

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)

class SimplifiedEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimplifiedEfficientNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            Swish()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(16, 24, expand_ratio=1, stride=1),
            MBConvBlock(24, 40, expand_ratio=6, stride=2),
            MBConvBlock(40, 80, expand_ratio=6, stride=2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(80, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=1, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(32 * 312 * 5, 128)  # 根据打印的形状调整输入大小
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 形状变为 (batch_size, 16, 625, 1)
        x = self.pool(F.relu(self.conv2(x)))  # 形状变为 (batch_size, 32, 312, 5)


        x = x.view(x.size(0), -1)  # 自动计算展平后的大小
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# 计算卷积输出大小的辅助函数
def calculate_conv_output_size(input_size, kernel_size, padding, stride):
    return (input_size - kernel_size + 2 * padding) // stride + 1

class LinearBinaryClassifier(nn.Module):
    def __init__(self, input_size=1250):
        super(LinearBinaryClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 2)  # 输出类别数为2
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x






class CustomShuffleNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomShuffleNetV2, self).__init__()
        # Load the pre-trained ShuffleNetV2 model
        self.shufflenet = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept input shape (1, 1, 1, 1250)
        self.shufflenet.conv1[0] = nn.Conv2d(1, 24, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False)
        
        # Modify the final fully connected layer for binary classification
        self.shufflenet.fc = nn.Linear(self.shufflenet.fc.in_features, num_classes)

    def forward(self, x):
        
        x = self.shufflenet(x)
        

        
        return x

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18Custom, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept (1, 1, 1250) input
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)
        
        # Modify the maxpool layer to handle (1, W) input
        self.resnet18.maxpool = nn.Identity()  # Remove maxpool layer
        
        # Adjust the fully connected layer for the desired number of output classes
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)
    

class BasicBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ShuffleNetV2Custom1d(nn.Module):
    def __init__(self, num_classes=2):
        super(ShuffleNetV2Custom1d, self).__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(24)
        self.relu = nn.ReLU(inplace=True)
        
        # 使用 BasicBlock1d 进行网络构建
        self.layer1 = self._make_layer(24, 48, 4, stride=2)
        self.layer2 = self._make_layer(48, 96, 4, stride=2)
        self.layer3 = self._make_layer(96, 192, 4, stride=2)
        self.layer4 = self._make_layer(192, 384, 4, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(384, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock1d(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock1d(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


'''
import torch.nn.functional as F

class Block(nn.Module):
#    Depthwise conv + Pointwise conv
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, in_planes, kernel_size=3,
                               stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv2 = nn.Conv1d(in_planes, out_planes,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class CNNModel(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2),
           512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, input_size, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.layers = self._make_layers(in_planes=32)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(1024, num_classes)

        # Calculate output size
        conv_output_size = self._calculate_conv_output_size(input_size, 3, 1, 1)
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            conv_output_size = self._calculate_conv_output_size(conv_output_size, 3, 1, stride)
        self.fc_input_size = 1024 * conv_output_size

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _calculate_conv_output_size(self, input_size, kernel_size, padding, stride):
        return (input_size - kernel_size + 2 * padding) // stride + 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

'''
