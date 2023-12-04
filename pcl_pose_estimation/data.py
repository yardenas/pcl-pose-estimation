import numpy as np
from pathlib import Path
from tensorflow import data as tfd


def normalize():
    pass


def collect_file_ids(data_path: str, split: float) -> tuple[list[int], list[int]]:
    files = list(Path(data_path).glob("label_data_*.npz"))
    num_files = len(files)
    ids = list(range(num_files))
    train_size = int(split * num_files)
    train = ids[:train_size]
    val = ids[train_size:]
    return train, val


def make_dataset(data_path: str, batch_size: int, split: float = 1.0) -> tfd.Dataset:
    def obs_and_labels(idx):
        obs = f"label_data_{idx}.npz"
        labels = f"label_data_{idx}.npz"
        return np.load(f"{data_path}/{obs}"), np.load(f"{data_path}/{labels}")

    file_indices = collect_file_ids(data_path, split)
    dataset = (
        tfd.Dataset.from_tensor_slices(file_indices)
        .map(obs_and_labels, num_parallel_calls=tfd.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tfd.AUTOTUNE)
    )
    return dataset
