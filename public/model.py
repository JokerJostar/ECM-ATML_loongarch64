import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=1000):
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
    

class ShuffleNetV2Custom(nn.Module):
    def __init__(self, num_classes=1000):
        super(ShuffleNetV2Custom, self).__init__()
        # Load the pretrained ShuffleNetV2 model
        self.shufflenet_v2 = models.shufflenet_v2_x1_0(pretrained=True)
        
        # Modify the first convolutional layer to accept (1, 1, 1250) input
        self.shufflenet_v2.conv1[0] = nn.Conv2d(1, 24, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)
        
        # Modify the maxpool layer to handle (1, W) input
        self.shufflenet_v2.maxpool = nn.Identity()  # Remove maxpool layer
        
        # Adjust the fully connected layer for the desired number of output classes
        self.shufflenet_v2.fc = nn.Linear(self.shufflenet_v2.fc.in_features, num_classes)

    def forward(self, x):
        return self.shufflenet_v2(x)


# 计算卷积输出大小的辅助函数
def calculate_conv_output_size(input_size, kernel_size, padding, stride):
    return (input_size - kernel_size + 2 * padding) // stride + 1




# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # 计算每层的输出大小
        conv1_output_size = calculate_conv_output_size(input_size, 3, 1, 1)
        pool1_output_size = conv1_output_size // 2
        conv2_output_size = calculate_conv_output_size(pool1_output_size, 3, 1, 1)
        pool2_output_size = conv2_output_size // 2
        conv3_output_size = calculate_conv_output_size(pool2_output_size, 3, 1, 1)
        pool3_output_size = conv3_output_size // 2
        conv4_output_size = calculate_conv_output_size(pool3_output_size, 3, 1, 1)
        pool4_output_size = conv4_output_size // 2
        conv5_output_size = calculate_conv_output_size(pool4_output_size, 3, 1, 1)
        pool5_output_size = conv5_output_size // 2
        conv6_output_size = calculate_conv_output_size(pool5_output_size, 3, 1, 1)
        pool6_output_size = conv6_output_size // 2

        self.fc1 = nn.Linear(1024 * pool6_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x



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