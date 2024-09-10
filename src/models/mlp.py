import torch.nn as nn
from torchvision.ops import MLP


class MLPHead(MLP):
    def __init__(
        self,
        in_channels,
        dim_hidden,
        num_layers=3,
        norm_layer=None,
        dropout=0.0,
    ):
        hidden_channels = [dim_hidden] * (num_layers - 1) + [1]
        super(MLPHead, self).__init__(
            in_channels, hidden_channels, inplace=False, norm_layer=norm_layer, dropout=dropout
        )


class mlp(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)
