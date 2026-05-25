import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        window_size,
        num_classes=4,
        conv_channels=[64, 128],
        kernel_sizes=[7, 5],
    ):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        conv_layers = []
        in_channels = 248
        current_time = window_size

        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            padding = kernel_size // 2
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )

            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(2))
            in_channels = out_channels
            current_time = current_time // 2

        self.conv = nn.Sequential(*conv_layers)
        flattened_size = in_channels * current_time

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x