import equinox as eqx
import jax
import jax.nn as jnn


class ConvBlock(eqx.Module):
    conv: eqx.nn.Conv3d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        *,
        key: jax.Array,
    ):
        self.conv = eqx.nn.Conv3d(in_channels, out_channels, kernel, stride=2, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class Encoder(eqx.Module):
    layers: list[ConvBlock]
    avg_pool: eqx.nn.AdaptiveAvgPool3d

    def __init__(
        self,
        num_layers: int,
        depth: int,
        in_channels: int,
        *,
        key: jax.Array,
    ):
        self.layers = []
        key, subkey = jax.random.split(key)
        for i in range(num_layers):
            out_channels = 2**i * depth
            self.layers.append(ConvBlock(in_channels, out_channels, 3, key=subkey))
            in_channels = out_channels
        self.avg_pool = eqx.nn.AdaptiveAvgPool3d(1)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = jnn.elu(layer(x))
        x = self.avg_pool(x).ravel()
        return x


class Model(eqx.Module):
    encoder: Encoder
    out0: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(
        self, in_channels: int, out_dim: int, layers: int = 4, *, key: jax.Array
    ):
        super().__init__()
        assert layers > 0
        encoder_key, out0_key, out_key = jax.random.split(key, 3)
        base_channels = 32
        self.encoder = Encoder(layers, base_channels, in_channels, key=encoder_key)
        out_layers = base_channels * 2 ** (layers - 1)
        self.out = eqx.nn.Linear(out_layers, out_layers, key=out_key)
        self.out0 = eqx.nn.Linear(out_layers, out_dim, key=out0_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.encoder(x)
        x = jnn.elu(self.out(x))
        x = self.out0(x)
        return x
