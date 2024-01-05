from typing import Any
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow import data as tfd


def load_data(data_path: str) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    data = np.load(data_path)
    obs, labels = data["obs"], data["labels"]
    return obs, labels


def make_dataset(
    data_path: str, batch_size: int, split: float = 1.0
) -> tfd.Dataset | tuple[tfd.Dataset, tfd.Dataset]:
    def normalize(x, mean, std):
        return (x - mean) / (std + 1e-8)

    def stats(labels):
        mean = np.mean(labels, axis=0)
        std = np.std(labels, axis=0)
        return mean, std

    obs, labels = load_data(data_path)
    num_object_types = obs.max() + 1
    mean, std = stats(labels)

    def process(obs, labels):
        obs = tf.one_hot(obs, num_object_types, axis=1)
        obs = tf.squeeze(obs, axis=2)
        labels = normalize(labels, mean, std)
        return obs, labels

    def dataset():
        return (
            tfd.Dataset.from_tensor_slices((obs, labels))
            .batch(batch_size)
            .shuffle(1000, seed=0)
            .map(process, num_parallel_calls=tfd.AUTOTUNE)
            .prefetch(tfd.AUTOTUNE)
        )

    train_dataset = dataset()
    if False:
        return train_dataset
    else:
        return train_dataset
