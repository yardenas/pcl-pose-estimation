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
    print("Stats:", mean, std)

    def process(obs, labels):
        obs = tf.one_hot(obs, num_object_types, axis=1)
        obs = tf.squeeze(obs, axis=2)
        labels = normalize(labels, mean, std)
        return obs, labels

    def dataset(obs, labels):
        return (
            tfd.Dataset.from_tensor_slices((obs, labels))
            .shuffle(1000, seed=0)
            .batch(batch_size)
            .map(process, num_parallel_calls=tfd.AUTOTUNE)
            .prefetch(tfd.AUTOTUNE)
        )

    def split_data():
        idx = int(len(obs) * split)
        train_obs, train_labels = obs[:idx], labels[:idx]
        val_obs, val_labels = obs[:idx], labels[:idx]
        return (train_obs, train_labels), (val_obs, val_labels)

    if split < 1.0:
        train_data, val_data = split_data()
        train_dataset = dataset(*train_data)
        val_dateset = dataset(*val_data)
        return train_dataset, val_dateset
    else:
        train_dataset = dataset(obs, labels)
        return train_dataset
