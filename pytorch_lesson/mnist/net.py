import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3)  # 26x26x64 -> 24x24x64
        self.pool1 = nn.MaxPool2d(2, 2)  # 24x24x64 -> # 12x12x64
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(64, 128, 3)  # 12x12x128 -> 10x10x128
        self.conv4 = nn.Conv2d(128, 256, 3) # 10x10x256 -> 8x8x256
        self.pool2 = nn.MaxPool2d(2, 2)  # 8x8x256 -> 4x4x256
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4*4*256, 128)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.dropout2(x)
        x = x.view(-1, 4*4*256)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
