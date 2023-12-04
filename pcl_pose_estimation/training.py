from typing import Iterator

import equinox as eqx
import jax
import optax
import numpy as np
import numpy.typing as npt
from pcl_pose_estimation.model import Model

DataGenerator = Iterator[Iterator[tuple[npt.ArrayLike, npt.ArrayLike]]]


@eqx.filter_value_and_grad
def compute_loss(model: Model, x: jax.Array, y: jax.Array) -> jax.Array:
    y_hat = jax.vmap(model)(x)
    return optax.l2_loss(y_hat, y).mean()


@eqx.filter_jit
def update_step(
    model: Model,
    x: jax.Array,
    y: jax.Array,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    loss, grads = compute_loss(model, x, y)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_model(
    model: Model,
    opt: optax.GradientTransformation,
    data_generator: DataGenerator,
):
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    for step, (x, y) in enumerate(data_generator):
        loss, model, opt_state = update_step(model, x, y, opt, opt_state)
        if step % 100 == 0:
            print(f"step={step}, loss={loss}")
    return model


def evaluate(
    model: Model,
    epoch_generator: DataGenerator,
):
    eval_data = [(x, y) for x, y in epoch_generator]
    x, y = jax.tree_map(np.stack, zip(*eval_data))
    y_hat = jax.vmap(model, x)
    return optax.l2_loss(y_hat, y).mean()
