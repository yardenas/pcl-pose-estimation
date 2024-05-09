import jax
import jax.nn as jnn
import jax.numpy as jnp


def estimate_variance_proxy(sample: jax.Array) -> float:
    """Estimates the variance proxy of a sub-Gaussian distribution

    Args:
        sample (jax.Array): A 2D array of shape (n_samples, n_features)

    Returns:
        float: An estimate of the variance proxy
    """
    mu = sample.mean(0)
    debiased_sample = sample - mu

    def f(lambda_):
        return (
            2.0 * jnn.logsumexp(jnp.tensordot(debiased_sample, lambda_, axes=1))
            - jnp.log(sample.shape[0])
        ) / (lambda_.dot(lambda_))

    return 0.0
