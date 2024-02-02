import equinox as eqx
import jax
import jax.nn as jnn


class VoxNet(eqx.Module):
    conv1: eqx.nn.Conv3d
    conv2: eqx.nn.Conv3d
    conv3: eqx.nn.Conv3d
    max_pool: eqx.nn.MaxPool3d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, input_channels: int, output_dim: int, *, key: jax.Array):
        super().__init__()
        conv1_key, conv2_key, conv3_key, linear1_key, linear2_key = jax.random.split(
            key, 5
        )
        self.conv1 = eqx.nn.Conv3d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(8, 8, 4),
            stride=2,
            key=conv1_key,
        )
        self.conv2 = eqx.nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 2),
            stride=2,
            key=conv2_key,
        )
        self.conv3 = eqx.nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 1),
            stride=(2, 2, 1),
            key=conv3_key,
        )
        self.max_pool = eqx.nn.MaxPool3d((4, 4, 2), 1)
        self.linear1 = eqx.nn.Linear(7 * 7 * 3 * 64, 128, key=linear1_key)
        self.linear2 = eqx.nn.Linear(128, output_dim, key=linear2_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnn.gelu(self.conv1(x))
        x = jnn.gelu(self.conv2(x))
        x = jnn.gelu(self.conv3(x))
        x = self.max_pool(x)
        x = x.ravel()
        x = jnn.gelu(self.linear1(x))
        x = self.linear2(x)
        return x
