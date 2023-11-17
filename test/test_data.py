import pytest
import numpy as np
from pcl_pose_estimation.data import get_data

DATA_SIZE = 32


@pytest.fixture
def mocked_data(monkeypatch):
    def mock(*args, **kwargs):
        x = np.random.normal(size=(DATA_SIZE, 100, 100, 20, 1))
        y = np.random.normal(size=(DATA_SIZE, 6))
        return dict(x=x, y=y)

    monkeypatch.setattr(np, "load", mock)


@pytest.mark.parametrize("train_split,expected", [(0.5, int(DATA_SIZE / 2)), (1.0, 0)])
def test_get_data(mocked_data, train_split, expected):
    _, (x_test, y_test) = get_data("dummy_path", train_split)
    assert x_test.shape[0] == expected
    assert y_test.shape[0] == expected
