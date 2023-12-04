import equinox as eqx
import jax


def count_params(model: eqx.Module):
    num_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    num_millions = num_params / 1000000
    print(f"Model # of parameters: {num_millions:.2f}M")
