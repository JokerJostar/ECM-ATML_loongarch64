import torch
import torch.nn as nn


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

device = torch.device("cpu")



# 加载模型
model = torch.load('D:/ECM_and_ATML/96-95/saved_net.pth',map_location=device)
model.to(device)
model.eval()

# 创建一个示例输入
example_input = torch.randn(1, 1, 1250).to(device)

# 转换为 TorchScript
traced_model = torch.jit.trace(model, example_input)
traced_model.save('D:/ECM_and_ATML/loongarch/saved_net.pt')

