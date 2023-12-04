import numpy as np


def epoch(arrays, batch_size, random_state):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    perm = random_state.permutation(indices)
    start = 0
    end = batch_size
    while end <= dataset_size:
        batch_perm = perm[start:end]
        yield tuple(array[batch_perm] for array in arrays)
        start = end
        end = start + batch_size


def epoch_generator(data, batch_size, num_epochs, seed):
    random_state = np.random.RandomState(seed)
    for _ in range(num_epochs):
        yield epoch(data, batch_size, random_state)


def get_data(path: str, train_split: float = 1.0):
    data = np.load(path)
    x, y = data["x"], data["y"]
    split = int(x.shape[0] * train_split)
    (x_train, x_test), (y_train, y_test) = map(lambda x: (x[:split], x[split:]), (x, y))
    return (x_train, y_train), (x_test, y_test)


def normalize():
    pass


def _make_dataset(data_path: str):
    pass
