from typing import Callable
import equinox as eqx
import jax
import optax
import numpy as np
from tensorflow import data as tfd
from pcl_pose_estimation.voxnet_model import VoxNet


@eqx.filter_value_and_grad
def compute_loss(
    model: VoxNet,
    x: jax.Array,
    y: jax.Array,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    y_hat = jax.vmap(model)(x)
    return loss_fn(y_hat, y).mean()


@eqx.filter_jit
def update_step(
    model: VoxNet,
    x: jax.Array,
    y: jax.Array,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
):
    loss, grads = compute_loss(model, x, y, loss_fn)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_model(
    model: VoxNet,
    opt: optax.GradientTransformation,
    data_generator: tfd.Dataset,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array] = optax.l2_loss,
) -> VoxNet:
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    for step, (x, y) in enumerate(data_generator.as_numpy_iterator()):
        loss, model, opt_state = update_step(model, x, y, opt, opt_state, loss_fn)
        if step % 100 == 0:
            print(f"step {step}: loss={loss:.4f}")
    return model


def evaluate(
    model: VoxNet,
    epoch_generator: tfd.Dataset,
    metric_fn: Callable[[jax.Array, jax.Array], jax.Array] = optax.l2_loss,
):
    eval_data = [(x, y) for x, y in epoch_generator]
    x, y = jax.tree_map(np.stack, zip(*eval_data))
    y_hat = jax.vmap(model, x)
    return metric_fn(y_hat, y).mean()
