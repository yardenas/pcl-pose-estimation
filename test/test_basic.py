import jax
import numpy as np
import pytest

from pcl_pose_estimation.voxnet_model import VoxNet
from pcl_pose_estimation.utils import count_params

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


def test_model():
    data = generate_test_data()
    x, y = data["x"], data["y"]
    in_channels = x.shape[1]
    output_dim = y.shape[-1]
    model = VoxNet(in_channels, output_dim, key=jax.random.PRNGKey(0))
    count_params(model)
    y = model(x[0])
    assert y.shape == (OUT_SIZE,)
