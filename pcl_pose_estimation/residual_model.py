# Base on https://github.com/paganpasta/eqxvision/blob/main/eqxvision/models/classification/resnet.py
import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jrandom


def _conv3x3(in_channels, out_channels, stride, dilation, key):
    return eqx.nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        use_bias=False,
        key=key,
    )


def _conv1x1(in_channels, out_channels, stride, key):
    return eqx.nn.Conv3d(
        in_channels, out_channels, kernel_size=1, stride=stride, use_bias=False, key=key
    )


class ResnetBlock(eqx.Module):
    conv1: eqx.nn.Conv3d
    conv2: eqx.nn.Conv3d
    conv3: eqx.nn.Conv3d | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        subsample: bool,
        dilation: int = 1,
        *,
        key: jrandom.KeyArray,
    ):
        keys = jrandom.split(key, 3)
        self.conv1 = _conv3x3(
            in_channels,
            out_channels,
            dilation=1 if subsample else 2,
            stride=2 if subsample else 1,
            key=keys[0],
        )
        self.conv2 = _conv3x3(out_channels, out_channels, 1, dilation, key=keys[1])
        self.conv3 = (
            _conv1x1(in_channels, out_channels, stride=2, key=keys[2])
            if subsample
            else None
        )

    def __call__(
        self, x: jax.Array, *, key: jrandom.KeyArray | None = None
    ) -> jax.Array:
        out = jnn.gelu(x)
        out = self.conv1(out)
        out = jnn.gelu(out)
        out = self.conv2(out)
        if self.conv3 is not None:
            x = jnn.gelu(x)
            x = self.conv3(x)
        return x + out


def _make_layer(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    dilation: int,
    *,
    key: jrandom.KeyArray,
) -> eqx.nn.Sequential:
    keys = jrandom.split(key, num_blocks)
    layers = [ResnetBlock(in_channels, out_channels, True, dilation, key=keys[0])] + [
        ResnetBlock(
            out_channels,
            out_channels,
            False,
            dilation,
            key=keys[i],
        )
        for i in range(1, num_blocks)
    ]
    return eqx.nn.Sequential(layers)


class Model(eqx.Module):
    layer0: eqx.nn.Conv3d
    max_pool: eqx.nn.MaxPool3d
    layer1: eqx.nn.Sequential
    layer2: eqx.nn.Sequential
    layer3: eqx.nn.Sequential
    layer4: eqx.nn.Sequential
    avg_pool: eqx.nn.AdaptiveAvgPool3d
    outs: eqx.nn.Linear

    def __init__(self, in_channels: int, out_dim: int, key: jrandom.KeyArray):
        keys = jrandom.split(key, 5)
        # TODO (yarden): figure out with Yunke the initial kernel size and their corresponding dimensions
        self.layer0 = eqx.nn.Conv3d(
            in_channels, 32, (5, 5, 2), dilation=(2, 2, 2), key=keys[0]
        )
        self.max_pool = eqx.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_layer(32, 64, 2, 2, key=keys[0])
        self.layer2 = _make_layer(64, 128, 2, 2, key=keys[1])
        self.layer3 = _make_layer(128, 256, 2, 2, key=keys[2])
        self.layer4 = _make_layer(256, 512, 2, 2, key=keys[3])
        self.avg_pool = eqx.nn.AdaptiveAvgPool3d(1)
        self.outs = eqx.nn.Linear(512, out_dim, key=keys[4])

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.layer0(x)
        x = jnn.gelu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x).ravel()
        return self.outs(x)
