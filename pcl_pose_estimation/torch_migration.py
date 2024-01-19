import json
import equinox as eqx
import jax

from pcl_pose_estimation.voxnet_model import VoxNet
from pcl_pose_estimation.voxnet_model_torch import PyTorchVoxNet


# https://docs.kidger.site/equinox/examples/serialisation/
def dump_voxnet(model: VoxNet, path: str, input_channels: int, output_dim: int) -> None:
    with open(path, "wb") as f:
        parameters = dict(in_channels=input_channels, output_dim=output_dim)
        serialized_config = json.dumps(parameters)
        f.write((serialized_config + "\n").encode())
        eqx.tree_deserialise_leaves(f, model)


def load_voxnet(path: str) -> VoxNet:
    with open(path, "rb") as f:
        parameters = json.loads(f.readline().decode())
        model = VoxNet(**parameters, key=jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(f, model)


def convert_to_torch(
    input_model: VoxNet, input_channels: int, output_dim: int
) -> PyTorchVoxNet:
    pytorch_model = PyTorchVoxNet(input_channels, output_dim)
    pytorch_model.state_dict().update(input_model)
    return pytorch_model
