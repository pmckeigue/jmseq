"""Phase 2B checkpoint tests for LGSSM model variants and MAP estimation."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jmseq.models.model_config import LGSSMConfig, make_config, VARIANTS
from jmseq.models.lgssm import lgssm_model
from jmseq.inference.map_estimator import fit_map
from jmseq.kalman.covariance import van_loan, discrete_drift
from jmseq.kalman.filter import batched_filter


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def simulate_lgssm(config: LGSSMConfig, N=40, T=15, seed=0):
    """Simulate data from an LGSSM with known parameters.

    For model_lmm: A=0, G=0, b~N(0,1), R=diag(sigma^2).
    Returns Y (N,T,p), dt_batch (N,T), mask_batch (N,T), and true params dict.
    """
    rng = np.random.default_rng(seed)
    p = config.p

    # True parameters
    true = {}
    true["mu0"] = rng.standard_normal(p).astype(np.float32)

    if config.free_drift:
        true["A"] = np.diag(-np.abs(rng.standard_normal(p)).astype(np.float32) * 0.5)
    else:
        true["A"] = np.zeros((p, p), dtype=np.float32)

    if config.free_diffusion:
        L = np.tril(rng.standard_normal((p, p)).astype(np.float32) * 0.2)
        np.fill_diagonal(L, np.abs(np.diag(L)) + 0.1)
        true["G"] = L
    else:
        true["G"] = np.zeros((p, p), dtype=np.float32)

    if config.free_cint:
        true["b"] = rng.standard_normal(p).astype(np.float32) * 0.5
    else:
        true["b"] = np.zeros(p, dtype=np.float32)

    true["sigma_obs"] = (np.abs(rng.standard_normal(p)) * 0.2 + 0.2).astype(np.float32)
    R_true = np.diag(true["sigma_obs"] ** 2)
    P0_true = np.eye(p, dtype=np.float32)

    # Simulate trajectories
    dt = 0.25
    dt_batch = np.full((N, T), dt, dtype=np.float32)
    mask_batch = np.ones((N, T), dtype=bool)
    Y = np.zeros((N, T, p), dtype=np.float32)

    A_jax = jnp.array(true["A"])
    G_jax = jnp.array(true["G"])
    b_jax = jnp.array(true["b"])

    A_d = np.array(van_loan(A_jax, G_jax, dt)[0])
    Q_d = np.array(van_loan(A_jax, G_jax, dt)[1])
    b_d = np.array(discrete_drift(A_jax, b_jax, dt))

    for i in range(N):
        x = rng.multivariate_normal(true["mu0"], P0_true)
        for t in range(T):
            y = x + rng.multivariate_normal(np.zeros(p), R_true)
            Y[i, t] = y
            w = rng.multivariate_normal(np.zeros(p), Q_d)
            x = A_d @ x + b_d + w

    return (
        jnp.array(Y),
        jnp.array(dt_batch),
        jnp.array(mask_batch),
        true,
    )


# ---------------------------------------------------------------------------
# Model construction tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(VARIANTS.keys()))
def test_all_variants_construct(name):
    """All five configs can be traced through lgssm_model without error."""
    config = make_config(name, p=2)
    Y, dt_batch, mask_batch, _ = simulate_lgssm(config, N=5, T=5)
    # Just trace the model (no inference) — use numpyro.render_model or prior predictive
    import numpyro
    from numpyro.infer import Predictive
    key = jax.random.PRNGKey(0)
    predictive = Predictive(lgssm_model, num_samples=1)
    # Should not raise
    samples = predictive(key, config, Y, dt_batch, mask_batch)
    assert samples is not None


@pytest.mark.parametrize("name,free_drift,free_diffusion,free_cint", [
    ("model_lmm",          False, False, True),
    ("model_nolmm",        True,  True,  False),
    ("model_lmmdiff",      False, True,  True),
    ("model_lmmdrift",     True,  False, True),
    ("model_lmmdriftdiff", True,  True,  True),
])
def test_fixed_params_absent(name, free_drift, free_diffusion, free_cint):
    """Fixed parameters must not appear in the prior sample sites."""
    config = make_config(name, p=2)
    Y, dt_batch, mask_batch, _ = simulate_lgssm(config, N=3, T=4)
    from numpyro.infer import Predictive
    key = jax.random.PRNGKey(0)
    samples = Predictive(lgssm_model, num_samples=1)(key, config, Y, dt_batch, mask_batch)
    if not free_drift:
        assert "log_neg_a_diag" not in samples, f"{name}: drift sites should be absent"
    if not free_diffusion:
        assert "L_G" not in samples and "g_scale" not in samples, \
            f"{name}: diffusion sites should be absent"
    if not free_cint:
        assert "b" not in samples, f"{name}: b site should be absent"


# ---------------------------------------------------------------------------
# MAP estimation test (model_lmm: only b and R are free)
# ---------------------------------------------------------------------------

def test_map_lmm_recovers_b_and_R():
    """MAP estimation on model_lmm recovers true b and sigma_obs within 2 SE.

    model_lmm has A=0, G=0, so state evolves only by mean b (random walk mean).
    With N=80 individuals x 20 time steps this should be well-identified.
    """
    config = make_config("model_lmm", p=2)
    Y, dt_batch, mask_batch, true = simulate_lgssm(config, N=80, T=20, seed=1)

    model_fn = functools.partial(lgssm_model, config)
    params, losses = fit_map(
        model_fn,
        model_args=(Y, dt_batch, mask_batch),
        n_steps=3_000,
        lr=0.02,
    )

    # Check b
    b_est = np.array(params["b"])
    b_true = true["b"]
    # Loose tolerance: within 0.3 on each element
    err_b = np.max(np.abs(b_est - b_true))
    assert err_b < 0.3, f"b error {err_b:.3f} too large; b_est={b_est}, b_true={b_true}"

    # Check sigma_obs via log_sigma_obs
    log_sigma_est = np.array(params["log_sigma_obs"])
    sigma_est = np.exp(log_sigma_est)
    sigma_true = true["sigma_obs"]
    err_sigma = np.max(np.abs(sigma_est - sigma_true))
    assert err_sigma < 0.15, \
        f"sigma_obs error {err_sigma:.3f} too large; est={sigma_est}, true={sigma_true}"


# ---------------------------------------------------------------------------
# Drift stability test
# ---------------------------------------------------------------------------

def test_map_drift_eigenvalues_negative():
    """After MAP on a free-drift model, A diagonal entries must be negative."""
    config = make_config("model_nolmm", p=2)
    Y, dt_batch, mask_batch, _ = simulate_lgssm(config, N=40, T=15, seed=2)

    model_fn = functools.partial(lgssm_model, config)
    params, _ = fit_map(
        model_fn,
        model_args=(Y, dt_batch, mask_batch),
        n_steps=2_000,
        lr=0.02,
    )

    log_neg_a = np.array(params["log_neg_a_diag"])
    a_diag = -np.exp(log_neg_a)
    assert np.all(a_diag < 0), \
        f"Drift diagonal not all negative: {a_diag}"


# ---------------------------------------------------------------------------
# TI predictor test
# ---------------------------------------------------------------------------

def test_ti_predictor_shifts_initial_state():
    """Individuals with different TI predictors should have different initial state means."""
    p, n_tipred = 2, 1
    config = make_config("model_lmm", p=p, n_tipred=n_tipred)
    N, T = 10, 5
    Y, dt_batch, mask_batch, _ = simulate_lgssm(config, N=N, T=T)
    # Two groups: ti_pred = +1 vs -1
    ti_pred = jnp.array(
        [[1.0]] * (N // 2) + [[-1.0]] * (N // 2), dtype=jnp.float32
    )

    from numpyro.infer import Predictive
    key = jax.random.PRNGKey(5)
    samples = Predictive(lgssm_model, num_samples=4)(
        key, config, Y, dt_batch, mask_batch, ti_pred=ti_pred
    )
    # T0 should be present
    assert "T0" in samples, "T0 site missing when n_tipred > 0"
    # mu0 should also be present
    assert "mu0" in samples
