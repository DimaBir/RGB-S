import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_of_channels: int = 10):
        super(Net, self).__init__()
        # CONV + MAXPOOL
        # 128x128
        self.conv1 = nn.Conv2d(num_of_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        # 32x32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        # 8x8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        # 2x2

        # Dropouts
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # FC
        self.fc1 = nn.Linear(256 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        # x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        # x = self.dropout1(x)
        # Flatten
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        """ Predicts the labels of a mini-batch of inputs
            @:param x: Input of NN
            @:return: Returns prediction for class with highest probability
            @:rtype: float
        """
        # x = F.softmax(self.forward(x), dim=1)
        m = nn.LogSoftmax(dim=1)
        x = m(self.forward(x))
        return x

    def save_networks(self, model_folder, optimizer, epoch):
        # Save models checkpoints
        torch.save({'optimizer_state_dict': optimizer.state_dict()}, model_folder)

        torch.save(self.state_dict(), model_folder)
        with open(model_folder, 'w+') as file:
            file.write(str(epoch + 1))




def create_model(num_of_channels: int):
    return Net(num_of_channels=num_of_channels)
