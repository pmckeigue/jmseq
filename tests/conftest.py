"""Shared fixtures for jmseq tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def poisson_glm_data():
    """Synthetic dataset for Poisson GLM.

    y ~ Poisson(exp(log_tobs + 0.5 + X @ true_beta))
    true_beta = [0.8, -0.5, 0.3]
    """
    rng = np.random.default_rng(42)
    N, P = 400, 3
    true_beta = np.array([0.8, -0.5, 0.3])
    X = rng.standard_normal((N, P))
    log_tobs = rng.uniform(-1.0, 0.0, size=N)  # intervals of 0.37–1 year
    beta0_true = 0.5
    log_mu = log_tobs + beta0_true + X @ true_beta
    y = rng.poisson(np.exp(log_mu)).astype(np.int32)
    return {
        "X": jnp.array(X),
        "y": jnp.array(y),
        "log_tobs": jnp.array(log_tobs),
        "true_beta": jnp.array(true_beta),
        "beta0_true": beta0_true,
        "N": N,
        "P": P,
    }
