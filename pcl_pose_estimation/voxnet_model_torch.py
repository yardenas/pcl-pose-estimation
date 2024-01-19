import torch
import torch.nn as nn


class PyTorchVoxNet(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(PyTorchVoxNet, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(8, 8, 4), stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(4, 4, 2), stride=2)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 1), stride=(2, 2, 1))
        self.max_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.linear1 = nn.Linear(64 * 4 * 4 * 4, 128)
        self.linear2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = torch.nn.functional.gelu(self.conv3(x))
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.gelu(self.linear1(x))
        x = self.linear2(x)
        return x
