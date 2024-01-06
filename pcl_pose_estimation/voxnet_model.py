import equinox as eqx
import jax
import jax.nn as jnn


class VoxNet(eqx.Module):
    conv1: eqx.nn.Conv3d
    conv2: eqx.nn.Conv3d
    max_pool: eqx.nn.AdaptiveMaxPool3d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, num_classes: int, input_size: int = 32, *, key: jax.Array):
        super().__init__()
        conv1_key, conv2_key, linear1_key, linear2_key = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv3d(
            in_channels=1, out_channels=32, kernel_size=5, stride=2, key=conv1_key
        )
        self.conv2 = eqx.nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=3, key=conv2_key
        )
        # Why 4? Because.
        target_size = input_size // 4
        self.max_pool = eqx.nn.AdaptiveAvgPool3d(target_size)
        self.linear1 = eqx.nn.Linear(target_size**3 * 32, 128, key=linear1_key)
        self.linear2 = eqx.nn.Linear(128, num_classes, key=linear2_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnn.gelu(self.conv1(x))
        x = jnn.gelu(self.conv2(x))
        x = self.max_pool(x)
        x = x.ravel()
        x = jnn.gelu(self.linear1(x))
        x = self.linear2(x)
        return x
