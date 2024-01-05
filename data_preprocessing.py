from pathlib import Path
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig


def collect_file_ids(
    data_path: str, split: float | None = None
) -> tuple[list[int], list[int]] | list[int]:
    files = list(Path(data_path).glob("label_data_*.npz"))
    num_files = len(files)
    assert num_files > 0
    ids = list(range(num_files))
    if split is None:
        return ids
    train_size = int(split * num_files)
    train = ids[:train_size]
    val = ids[train_size:]
    return train, val


def obs_and_labels(data_path, idx):
    obs = f"obs_data_{idx}.npz"
    labels = f"label_data_{idx}.npz"
    obs_data = np.load(f"{data_path}/{obs}")
    labels_data = np.load(f"{data_path}/{labels}")
    return obs_data["arr_0"], labels_data["arr_0"][..., :3]


def load_data(data_path: str) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    print(f"Loading data from {data_path}")
    ids = collect_file_ids(data_path)
    all_obs, all_labels = [], []
    for id in ids:
        obs, labels = obs_and_labels(data_path, id)
        all_obs.append(obs)
        all_labels.append(labels)
    all_obs = np.concatenate(all_obs)
    all_labels = np.concatenate(all_labels)
    return obs, labels


def clean_duplicates(
    obs: npt.NDArray[Any], labels: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    before = len(obs)
    unique_labels, ids = np.unique(labels, return_index=True, axis=0)
    unique_obs = obs[ids]
    after = len(unique_obs)
    print(f"Removed {before - after} duplicates")
    return unique_obs, unique_labels


def shuffle_data(
    obs: npt.NDArray[Any], labels: npt.NDArray[Any], seed: int
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    idx = np.random.RandomState(seed).permutation(len(obs))
    return obs[idx], labels[idx]


def write_data(obs: npt.NDArray[Any], labels: npt.NDArray[Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=obs, labels=labels)


@hydra.main(config_name="preprocessing", config_path=".", version_base=None)
def main(config: DictConfig) -> None:
    obs, labels = load_data(config.path)
    obs, labels = clean_duplicates(obs, labels)
    obs, labels = shuffle_data(obs, labels, config.seed)
    write_data(obs, labels, config.out_path)
    print("Done!")


if __name__ == "__main__":
    main()
