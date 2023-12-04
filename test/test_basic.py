import jax
import equinox as eqx
import numpy as np
import pytest

from pcl_pose_estimation.residual_model import Model

DATA_SIZE = 32
OUT_SIZE = 6


def generate_test_data():
    x = np.random.normal(size=(DATA_SIZE, 1, 100, 100, 20))
    y = np.random.normal(size=(DATA_SIZE, OUT_SIZE))
    return dict(x=x, y=y)


@pytest.fixture
def mocked_data(monkeypatch):
    def mock(*args, **kwargs):
        return generate_test_data()

    monkeypatch.setattr(np, "load", mock)


def count_params(model: eqx.Module):
    num_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    num_millions = num_params / 1000000
    print(f"Model # of parameters: {num_millions:.2f}M")


def test_model():
    data = generate_test_data()
    x, y = data["x"], data["y"]
    in_channels = x.shape[1]
    output_dim = y.shape[-1]
    model = Model(
        in_channels,
        output_dim,
        key=jax.random.PRNGKey(0),
    )
    count_params(model)
    y = model(x[0])
    assert y.shape == (OUT_SIZE,)
