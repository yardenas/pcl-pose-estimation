import hydra
import optax
import jax
import equinox as eqx
from omegaconf import DictConfig

from pcl_pose_estimation.data import make_dataset
from pcl_pose_estimation.voxnet_model import VoxNet
from pcl_pose_estimation.training import evaluate, train_model
from pcl_pose_estimation.utils import count_params


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(config: DictConfig) -> None:
    train_data, val_data = make_dataset(
        config.data.path, config.training.batch_size, config.data.train_split
    )
    example_x, example_y = next(iter(train_data))
    assert example_x.ndim == 5
    output_dim = example_y.shape[-1]
    in_channels = example_x.shape[1]
    model = VoxNet(in_channels, output_dim, key=jax.random.PRNGKey(0))
    count_params(model)
    opt = optax.adam(config.training.learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    for _ in range(config.training.epochs):
        model, opt_state = train_model(model, opt, opt_state, train_data)
        result = evaluate(model, val_data)
        print(f"Evaluation result: {result}")


if __name__ == "__main__":
    main()
