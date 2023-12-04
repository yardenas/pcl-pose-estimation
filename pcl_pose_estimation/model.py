from typing import Sequence
import equinox as eqx
import jax
import jax.nn as jnn
import numpy as np


class ConvBlock(eqx.Module):
    conv: eqx.nn.Conv3d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        *,
        key: jax.random.PRNGKey,
    ):
        self.conv = eqx.nn.Conv3d(in_channels, out_channels, kernel, stride=2, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x.transpose((3, 0, 1, 2))).transpose((1, 2, 3, 0))


class Encoder(eqx.Module):
    layers: list[ConvBlock]

    def __init__(
        self,
        kernels: list[int],
        depth: int,
        in_channels: int,
        *,
        key: jax.random.PRNGKey,
    ):
        self.layers = []
        key, subkey = jax.random.split(key)
        for i, kernel in enumerate(kernels):
            out_channels = 2**i * depth
            self.layers.append(ConvBlock(in_channels, out_channels, kernel, key=subkey))
            in_channels = out_channels

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = jnn.gelu(layer(x))
        return x


class Decoder(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(
        self,
        layers: list[int],
        input_dim: int,
        *,
        key: jax.random.PRNGKey,
    ):
        self.layers = []
        key, subkey = jax.random.split(key)
        for linear_layer in layers:
            self.layers.append(eqx.nn.Linear(input_dim, linear_layer, key=subkey))
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
        input_shape: Sequence[int],
        output_dim: int,
        kernels: list[int],
        depth: int,
        in_channels: int,
        linear_layers: list[int],
        *,
        key: jax.random.PRNGKey,
    ):
        super().__init__()
        self.norm = eqx.nn.LayerNorm(in_channels)
        encoder_key, decoder_key, out_key = jax.random.split(key, 3)
        self.encoder = Encoder(kernels, depth, in_channels, key=encoder_key)
        dummy_x = jax.random.normal(jax.random.PRNGKey(0), (in_channels, *input_shape))
        dummy_y = self.encoder(dummy_x)
        input_dim = np.prod(dummy_y.shape)
        self.decoder = Decoder(linear_layers, input_dim, key=decoder_key)
        self.out = eqx.nn.Linear(
            self.decoder.layers[-1].out_features, output_dim, key=out_key
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.norm(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out(x)
        return x
