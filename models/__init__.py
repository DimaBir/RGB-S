import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_of_channels: int = 10):
        super(Net, self).__init__()
        # 128x128
        self.conv1 = nn.Conv2d(num_of_channels, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 32x32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # 8x8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # 2x2
        self.fc1 = nn.Linear(256 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_networks(self, model_folder, optimizer, epoch):
        # Save models checkpoints
        torch.save({'optimizer_state_dict': optimizer.state_dict()}, model_folder)

        torch.save(self.state_dict(), model_folder)
        with open(model_folder, 'w+') as file:
            file.write(str(epoch + 1))

    def predict(self, x):
        """ Predicts the labels of a mini-batch of inputs
            @:param x: Input of NN
            @:return: Returns prediction for class with highest probability
            @:rtype: float
        """
        x = F.softmax(self.forward(x), dim=1)
        return x


def create_model(num_of_channels: int):
    return Net(num_of_channels=num_of_channels)
