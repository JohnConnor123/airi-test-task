import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf


current_dir = os.getcwd()
os.chdir(os.path.dirname(os.sep.join(__file__.split(os.sep)[:-2])))
cfg = OmegaConf.load("src/config/config.yaml")
os.chdir(current_dir)


class CNN(nn.Module):
    def __init__(self, input_size=358, out_channels=500):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, out_channels=out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, 500, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(500)
        self.conv3 = nn.Conv1d(500, 250, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(250)
        self.conv4 = nn.Conv1d(250, 100, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(100)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.activation(x)
        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.activation(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = CNN()
    print("Total number of parameters:", sum(p.numel() for p in model.parameters()))

    X_train = torch.rand(32, 358)
    print(model(X_train))
