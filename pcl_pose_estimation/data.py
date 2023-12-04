import numpy as np
from pathlib import Path
from tensorflow import data as tfd
import tensorflow as tf


def collect_file_ids(data_path: str, split: float) -> tuple[list[int], list[int]]:
    files = list(Path(data_path).glob("label_data_*.npz"))
    num_files = len(files)
    assert num_files > 0
    ids = list(range(num_files))
    train_size = int(split * num_files)
    train = ids[:train_size]
    val = ids[train_size:]
    return train, val


def make_dataset(
    data_path: str, batch_size: int, split: float = 1.0
) -> tfd.Dataset | tuple[tfd.Dataset, tfd.Dataset]:
    def obs_and_labels(idx):
        obs = f"obs_data_{idx}.npz"
        labels = f"label_data_{idx}.npz"
        obs_data = np.load(f"{data_path}/{obs}")
        labels_data = np.load(f"{data_path}/{labels}")
        return obs_data["arr_0"], labels_data["arr_0"]

    def datagen(indices):
        for idx in indices:
            yield obs_and_labels(idx)

    train, val = collect_file_ids(data_path, split)
    example = obs_and_labels(0)
    output_shape = tuple(map(lambda x: x.shape, example))
    num_object_types = example[0].max()

    def one_hot(obs, labels):
        obs = tf.one_hot(obs, num_object_types, axis=1)
        obs = tf.squeeze(obs, axis=2)
        return obs, labels

    def dataset(indices):
        return (
            tfd.Dataset.from_generator(
                lambda: datagen(indices),
                output_types=(tf.uint8, tf.float32),
                output_shapes=output_shape,
            )
            .rebatch(batch_size)
            .shuffle(100, seed=0)
            .map(one_hot, num_parallel_calls=tfd.AUTOTUNE)
            .prefetch(tfd.AUTOTUNE)
        )

    train_dataset = dataset(train)
    if len(val) > 0:
        val_dataset = dataset(val)
        return train_dataset, val_dataset
    else:
        return train_dataset
