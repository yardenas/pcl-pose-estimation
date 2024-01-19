import torch
import torch.nn as nn

class PyTorchVoxNet(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(PyTorchVoxNet, self).__init__()

        # PyTorch equivalent layers
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

# Load trained parameters from Equinox VoxNet
equinox_params = ...  # Load your Equinox model parameters here

# Instantiate PyTorch model
pytorch_model = PyTorchVoxNet(input_channels=your_input_channels, output_dim=your_output_dim)

# Convert and load parameters
with torch.no_grad():
    pytorch_model.conv1.weight.copy_(torch.tensor(equinox_params["conv1"]["weights"]))
    pytorch_model.conv1.bias.copy_(torch.tensor(equinox_params["conv1"]["bias"]))

    pytorch_model.conv2.weight.copy_(torch.tensor(equinox_params["conv2"]["weights"]))
    pytorch_model.conv2.bias.copy_(torch.tensor(equinox_params["conv2"]["bias"]))

    pytorch_model.conv3.weight.copy_(torch.tensor(equinox_params["conv3"]["weights"]))
    pytorch_model.conv3.bias.copy_(torch.tensor(equinox_params["conv3"]["bias"]))

    pytorch_model.linear1.weight.copy_(torch.tensor(equinox_params["linear1"]["weights"]))
    pytorch_model.linear1.bias.copy_(torch.tensor(equinox_params["linear1"]["bias"]))

    pytorch_model.linear2.weight.copy_(torch.tensor(equinox_params["linear2"]["weights"]))
    pytorch_model.linear2.bias.copy_(torch.tensor(equinox_params["linear2"]["bias"]))

# Set your PyTorch model to evaluation mode
pytorch_model.eval()

# Now you can use pytorch_model for inference in PyTorch