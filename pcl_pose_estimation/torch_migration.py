from collections import OrderedDict

import numpy as np
import torch

from pcl_pose_estimation.voxnet_model import VoxNet
from pcl_pose_estimation.voxnet_model_torch import PyTorchVoxNet


def convert_to_torch(
    input_model: VoxNet, input_channels: int, output_dim: int
) -> PyTorchVoxNet:
    pytorch_model = PyTorchVoxNet(input_channels, output_dim)
    state_dict = OrderedDict()
    for name in ["conv1", "conv2", "conv3", "linear1", "linear2"]:
        leaf = getattr(input_model, name)
        state_dict[f"{name}.weight"] = torch.from_numpy(np.asarray(leaf.weight))
        state_dict[f"{name}.bias"] = torch.from_numpy(np.asarray(leaf.bias).flatten())
    assert len(pytorch_model.state_dict()) == len(state_dict)
    load_result = pytorch_model.load_state_dict(state_dict)
    assert len(load_result.missing_keys) == len(load_result.missing_keys) == 0
    return pytorch_model
