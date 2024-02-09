import equinox as eqx
import hydra
import jax
import optax
from omegaconf import DictConfig

from pcl_pose_estimation.data import make_dataset
from pcl_pose_estimation.io import dump_voxnet
from pcl_pose_estimation.training import evaluate, train_model
from pcl_pose_estimation.utils import count_params
from pcl_pose_estimation.voxnet_model import VoxNet


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(config: DictConfig) -> None:
    train_data, val_data = make_dataset(
        config.data.path, config.training.batch_size, config.data.train_split
    )
    example_x, example_y = next(iter(train_data))
    assert example_x.ndim == 5
    output_dim = example_y.shape[-1]
    in_channels = example_x.shape[1]
    print(
        f"Starting training...\nOutput dimension: {output_dim}, input channels: {in_channels}"
    )
    model = VoxNet(in_channels, output_dim, key=jax.random.PRNGKey(0))
    count_params(model)
    opt = optax.adam(config.training.learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    for i in range(config.training.epochs):
        model, opt_state = train_model(model, opt, opt_state, train_data)
        result = evaluate(model, val_data)
        print(f"Evaluation result: {result}")
        dump_voxnet(model, f"{config.training.checkpoint_path}_{i}.ckpt", in_channels, output_dim)


if __name__ == "__main__":
    main()
