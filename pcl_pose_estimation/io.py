import json

import equinox as eqx
import jax

from pcl_pose_estimation.voxnet_model import VoxNet


# https://docs.kidger.site/equinox/examples/serialisation/
def dump_voxnet(model: VoxNet, path: str, input_channels: int, output_dim: int) -> None:
    with open(path, "wb") as f:
        parameters = dict(input_channels=input_channels, output_dim=output_dim)
        serialized_config = json.dumps(parameters)
        f.write((serialized_config + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_voxnet(path: str) -> tuple[VoxNet, dict[str, int]]:
    with open(path, "rb") as f:
        parameters = json.loads(f.readline().decode())
        model = VoxNet(**parameters, key=jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(f, model), parameters
