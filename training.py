from typing import Iterable
import equinox as eqx
import jax
import optax

from model import Model


@eqx.filter_value_and_grad
def compute_loss(model: Model, x: jax.Array, y: jax.Array) -> jax.Array:
    y_hat = jax.vmap(model, x)
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
    steps: int,
    data_loader: Iterable[tuple[jax.Array, jax.Array]],
):
    opt_state = opt.init(model)
    for step, (x, y) in zip(range(steps), data_loader):
        loss, model, opt_state = update_step(model, x, y, opt, opt_state)
        if step % 100 == 0:
            print(f"step={step}, loss={loss}")
    return model
