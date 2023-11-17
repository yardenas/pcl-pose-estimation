import hydra
import optax
from omegaconf import DictConfig

from pcl_pose_estimation.data import get_data
from pcl_pose_estimation.model import Model
from pcl_pose_estimation.training import evaluate, train_model


@hydra.main(config_name="config")
def main(config: DictConfig):
    train_data, test_data = get_data(config.data.path)
    example_x, example_y = train_data[0]
    assert example_x.ndim == 5
    input_dim = example_x.shape[1]
    output_dim = example_y.shape[-1]
    in_channels = example_x.shape[-1]
    model = Model(
        input_dim,
        output_dim,
        config.model.kernels,
        config.model.depth,
        in_channels,
        config.model.linear_layers,
    )
    opt = optax.adam(config.training.learning_rate)
    model = train_model(model, opt, config.training.epochs, train_data)
    result = evaluate(model, test_data)
    print(f"Evaluation result: {result}")


if __name__ == "__main__":
    main()
