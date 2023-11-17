import equinox as eqx
import jax
import jax.nn as jnn
import numpy as np


class ResidualConvBlock(eqx.Module):
    conv: eqx.nn.Conv3d

    def __init__(
        self, in_channels: int, out_channels: int, kernel: tuple[int, int, int]
    ):
        self.conv = eqx.nn.Conv3d(in_channels, out_channels, kernel, stride=2)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x) + x


class Encoder(eqx.Module):
    layers: list[ResidualConvBlock]

    def __init__(
        self,
        kernels: list[int],
        depth: int,
        in_channels: int,
    ):
        self.layers = []
        for i, kernel in enumerate(kernels):
            out_channels = i**2 * depth
            self.encoder.append(
                eqx.nn.Conv3d(in_channels, out_channels, kernel, stride=2)
            )
            in_channels = out_channels

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = jnn.gelu(layer(x))
        return x


class Decoder(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, layers: list[int], input_dim: int):
        for linear_layer in layers:
            self.layers.append(eqx.nn.Linear(input_dim, linear_layer))
            input_dim = linear_layer

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = jnn.gelu(layer(x))
        return x


class Model(eqx.Module):
    norm: eqx.nn.LayerNorm
    encoder: Encoder
    decoder: Decoder
    out: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernels: list[int],
        depth: int,
        in_channels: int,
        linear_layers: list[int],
    ):
        super().__init__()
        self.norm = eqx.nn.LayerNorm(in_channels)
        self.encoder = Encoder(kernels, depth, in_channels)
        dummy_x = jax.random.normal(
            jax.random.PRNGKey(0), (input_dim, input_dim, input_dim, in_channels)
        )
        dummy_y = self.encoder(dummy_x)
        input_dim = np.prod(dummy_y.shape)
        self.decoder = Decoder(linear_layers, input_dim)
        self.out = eqx.nn.Linear(self.decoder.layers[-1].out_features, output_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.norm(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out(x)
        return x
