import jax
import jax.nn as jnn
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from jaxopt._src.lbfgs import LbfgsState
from scipy import stats

jax.config.update("jax_enable_x64", True)


def estimate_variance_proxy(
    sample: jax.Array, initial_guess: jax.Array, tol: float = 0.001
) -> tuple[float, LbfgsState]:
    """Estimates the variance proxy of a sub-Gaussian distribution

    Args:
        sample (jax.Array): A 2D array of shape (n_samples, n_features)

    Returns:
        float: An estimate of the variance proxy
    """
    if sample.ndim != 2:
        raise ValueError("Sample must be a 2D array")
    if initial_guess.shape[0] != sample.shape[1]:
        raise ValueError(
            "Initial guess must have the same number of features as the sample"
        )
    mu = sample.mean(0)
    debiased_sample = sample - mu

    def f(lambda_):
        return (
            -2.0
            * (
                jnn.logsumexp(jnp.tensordot(debiased_sample, lambda_, axes=1))
                - jnp.log(sample.shape[0])
            )
            / (lambda_.dot(lambda_))
        )

    solver = jaxopt.LBFGS(fun=f, tol=tol)
    _, state = solver.run(initial_guess)
    return -state.value, state


# Generate random samples from a normal distribution
np.random.seed(0)
norm_samples = np.random.normal(loc=0, scale=2, size=1000)

# Generate random samples from a uniform distribution
uniform_bound = np.sqrt(3)
uniform_samples = np.random.uniform(low=-uniform_bound, high=uniform_bound, size=1000)

x_norm = np.linspace(-3, 3, 1000)
cdf_norm = stats.norm.cdf(x_norm, loc=0, scale=1)

x_uniform = np.linspace(-uniform_bound, uniform_bound, 1000)
cdf_uniform = stats.uniform.cdf(x_uniform, loc=-uniform_bound, scale=2 * uniform_bound)


def mfg_bound(sample, c4):
    return jnp.mean(jnp.exp((sample / c4) ** 2))


c4_values = np.linspace(0.9, 10, 100)
bounds = np.asarray([mfg_bound(x_uniform, c4) for c4 in c4_values]) - 2.0
# Using the fact that `mfg_bound` is monotonically decreasing w.r.t c4, find the first c4
# value that causes bounds to be non-negative
best_value = c4_values[jnp.argmin(bounds > 0)]

hoeffdings_bound = (2 * uniform_bound) ** 2 / 4

best_value_2, state = estimate_variance_proxy(
    uniform_samples[:, None], best_value[None]
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

ax1.plot(c4_values, bounds, color="black")
ax1.plot(best_value, 0.0, marker="*", color="tab:blue", label="Best C4 Value for MFG")
ax1.plot(
    best_value_2,
    0.0,
    marker="*",
    color="tab:orange",
    label="Best Value Optimization-based",
)
ax1.plot(1.0, 0.0, marker="*", markersize=1.5, color="tab:green", label="Correct Value")
ax1.plot(
    hoeffdings_bound, 0.0, marker="*", color="tab:purple", label="Hoeffding's Bound"
)
ax1.axhline(y=0.0, color="tab:red", linestyle="--")
ax1.text(
    best_value + 0.05,
    0.0 + 0.05,
    f"({round(best_value, 3)})",
    verticalalignment="bottom",
    horizontalalignment="left",
)
ax1.set_xlabel("C4 Value")
ax1.set_ylabel("MFG Bound")
ax1.grid(True)
ax1.legend(loc="upper right")

# CDFs on the right
cdf_norm_c4 = stats.norm.cdf(x_norm, loc=0, scale=best_value)
cdf_norm_heoffding = stats.norm.cdf(x_norm, loc=0, scale=hoeffdings_bound)
ax2.plot(x_norm, cdf_norm_c4, color="tab:blue", label="Normal CDF C4")
ax2.plot(x_norm, cdf_norm_heoffding, color="tab:purple", label="Normal CDF Hoeffding's")
ax2.plot(x_norm, cdf_norm, color="tab:green", label="Normal CDF")
ax2.plot(x_uniform, cdf_uniform, color="tab:orange", label="Uniform CDF")
ax2.set_ylabel("CDF")
ax2.legend(loc="lower right")

plt.show()
