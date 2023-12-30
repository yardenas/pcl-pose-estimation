import hydra
import optax
import jax
from omegaconf import DictConfig

from pcl_pose_estimation.data import make_dataset
from pcl_pose_estimation.model import Model
from pcl_pose_estimation.training import evaluate, train_model
from pcl_pose_estimation.utils import count_params


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(config: DictConfig):
    train_data, val_data = make_dataset(
        config.data.path, config.training.batch_size, config.data.train_split
    )
    example_x, example_y = next(iter(train_data))
    train_data = train_data.repeat(config.training.epochs)
    assert example_x.ndim == 5
    output_dim = example_y.shape[-1]
    in_channels = example_x.shape[1]
    model = Model(
        in_channels,
        output_dim,
        config.model.layers,
        key=jax.random.PRNGKey(0),
    )
    count_params(model)
    opt = optax.adam(config.training.learning_rate)
    model = train_model(model, opt, train_data)
    result = evaluate(model, val_data)
    print(f"Evaluation result: {result}")


if __name__ == "__main__":
    main()
