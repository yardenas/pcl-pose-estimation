import pathlib
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import torch

from pcl_pose_estimation.voxnet_model import VoxNet
from pcl_pose_estimation.torch_migration import (
    dump_voxnet,
    load_voxnet,
    convert_to_torch,
)

INPUT_CHANNELS = 1
OUTPUT_DIMS = 6
DUMP_MODEL = "voxnet.bin"
RANDOM_KEY = jax.random.PRNGKey(0)
BATCH_SIZE = 300


@pytest.fixture
def jax_dump_model():
    model = VoxNet(INPUT_CHANNELS, OUTPUT_DIMS, key=RANDOM_KEY)
    dump_voxnet(model, DUMP_MODEL, INPUT_CHANNELS, OUTPUT_DIMS)
    yield
    pathlib.Path(DUMP_MODEL).unlink()


@pytest.fixture
def random_inputs():
    key = jax.random.PRNGKey(1)
    inputs = jax.random.normal(key, (BATCH_SIZE, INPUT_CHANNELS, 100, 100, 20))
    return inputs


@pytest.fixture
def jax_load_model(jax_dump_model):
    model = load_voxnet(DUMP_MODEL)
    return model


@pytest.fixture
def pytorch_load_model(jax_load_model, jax_dump_model):
    voxnet_torch = convert_to_torch(jax_load_model, INPUT_CHANNELS, OUTPUT_DIMS)
    return voxnet_torch


def test_forward_pass(jax_load_model, pytorch_load_model, random_inputs):
    jax_output = jax.vmap(jax_load_model)(random_inputs)
    jax_output_np = np.asarray(jax_output)
    random_inputs_torch = torch.tensor(np.asarray(random_inputs))
    pytorch_output = pytorch_load_model(random_inputs_torch)
    pytorch_output_np = pytorch_output.detach().numpy()
    assert jnp.allclose(jax_output_np, pytorch_output_np, rtol=1e-4, atol=1e-4)
