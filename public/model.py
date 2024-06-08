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



# 计算卷积输出大小的辅助函数
def calculate_conv_output_size(input_size, kernel_size, padding, stride):
    return (input_size - kernel_size + 2 * padding) // stride + 1


# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_size=1250):
        super(CNNModel, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1))
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1))
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(1, 3), padding=(0, 1))
        self.pool = nn.MaxPool2d((1, 2), (1, 2))
        self.dropout = nn.Dropout(0.5)

        # 计算每层的输出大小
        pool6_output_size = input_size // (2 ** 6)  # 经过6次池化，每次池化尺寸减半

        self.fc1 = nn.Linear(1024 * pool6_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dequant(x)
        return x



class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleNetV2Block, self).__init__()
        self.stride = stride
        mid_channels = out_channels // 2

        if self.stride == 1:
            assert in_channels % 2 == 0
            self.branch_main = nn.Sequential(
                nn.Conv2d(in_channels // 2, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels)
            )
        else:
            self.branch_main = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, out_channels - in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels - in_channels)
            )
            self.branch_proj = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(in_channels)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat((x1, self.branch_main(x2)), dim=1)
        else:
            out = torch.cat((self.branch_proj(x), self.branch_main(x)), dim=1)

        out = self.channel_shuffle(out)
        out = F.relu(out, inplace=True)
        return out

    def channel_shuffle(self, x):
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, 2, num_channels // 2, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(ShuffleNetV2, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )

        self.stage2 = self._make_stage(4, 8, 1)   # 从1个block减少到1个
        self.stage3 = self._make_stage(8, 16, 1)  # 从1个block减少到1个

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [ShuffleNetV2Block(in_channels, out_channels, stride=2)]
        for _ in range(1, num_blocks):
            layers.append(ShuffleNetV2Block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
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