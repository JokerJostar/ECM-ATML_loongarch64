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

class OptimizedAFNet(nn.Module):
    def __init__(self):
        super(OptimizedAFNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 1), stride=(2, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 1), stride=(2, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1), stride=(2, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)

        pooled_output = self.global_avg_pool(conv5_output)
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



class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleNetV2Block, self).__init__()
        self.stride = stride

        mid_channels = out_channels // 2

        if self.stride == 1:
            self.branch_main = nn.Sequential(
                nn.Conv2d(in_channels // 2, mid_channels, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch_main = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=stride, padding=2, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.branch_proj = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=stride, padding=2, bias=False),
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
            nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )

        self.stage2 = self._make_stage(4, 8, 4)  # Increase number of blocks
        self.stage3 = self._make_stage(8, 16, 2)  # Increase number of blocks

        self.fc = nn.Linear(16, num_classes)

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
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.fc(x)
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
