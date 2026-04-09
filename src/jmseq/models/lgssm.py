"""
NumPyro LGSSM model with five variants controlled by LGSSMConfig.

Continuous-time SDE:
  dX(t) = (A X(t) + b) dt + G dW(t)

Observation model (LAMBDA = I_p):
  y_t = x_t + v_t,   v_t ~ N(0, R)

Initial state:
  x_0 ~ N(m0_i, P0)   where m0_i = mu0 + T0 @ ti_pred_i

Parameter priors
----------------
A  (drift, if free)  : diagonal entries parameterised as -exp(log_neg_a_diag)
                       (ensures negative real parts = stability);
                       off-diagonal entries ~ Normal(0, 1).
G  (diffusion, if free): lower-triangular Cholesky L_G with
                         LowerCholesky constraint; GGT = L_G L_G^T.
b  (CINT, if free)   : Normal(0, 1), length p.
R  (obs noise)       : diagonal, entries exp(log_sigma_obs)^2.
mu0                  : Normal(0, 5), length p.
T0 (TI pred effect)  : Normal(0, 1), shape (p, n_tipred) — omitted if n_tipred=0.
L_P0 (init cov)      : LowerCholesky, P0 = L_P0 L_P0^T.

The log-likelihood is injected as a NumPyro factor via the JAX Kalman filter.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

from jmseq.models.model_config import LGSSMConfig
from jmseq.kalman.filter import batched_filter


def lgssm_model(
    config: LGSSMConfig,
    Y: jnp.ndarray,
    dt_batch: jnp.ndarray,
    mask_batch: jnp.ndarray,
    ti_pred: jnp.ndarray | None = None,
):
    """NumPyro LGSSM model.

    Parameters
    ----------
    config      : LGSSMConfig specifying which parameters are free.
    Y           : (N, T, p)  padded observations (values at masked steps are ignored).
    dt_batch    : (N, T)     padded time intervals.
    mask_batch  : (N, T)     bool — True where a real observation exists.
    ti_pred     : (N, n_tipred) time-invariant predictors, or None.
    """
    p = config.p
    N = Y.shape[0]

    # ------------------------------------------------------------------ #
    # Drift matrix A
    # ------------------------------------------------------------------ #
    if config.free_drift:
        # Diagonal: force negative via -exp(.)
        log_neg_a_diag = numpyro.sample(
            "log_neg_a_diag", dist.Normal(jnp.zeros(p), jnp.ones(p))
        )
        a_diag = -jnp.exp(log_neg_a_diag)           # (p,) all negative
        if p > 1:
            # Off-diagonal entries (row-major, upper then lower triangle)
            n_offdiag = p * (p - 1)
            a_offdiag = numpyro.sample(
                "a_offdiag", dist.Normal(jnp.zeros(n_offdiag), jnp.ones(n_offdiag))
            )
            A = jnp.diag(a_diag)
            idx = [(i, j) for i in range(p) for j in range(p) if i != j]
            for k, (i, j) in enumerate(idx):
                A = A.at[i, j].set(a_offdiag[k])
        else:
            A = jnp.diag(a_diag)
    else:
        A = jnp.zeros((p, p))

    # ------------------------------------------------------------------ #
    # Diffusion matrix G (lower-triangular Cholesky)
    # ------------------------------------------------------------------ #
    if config.free_diffusion:
        L_G = numpyro.sample(
            "L_G",
            dist.LKJCholesky(p, concentration=1.0)
            if p > 1
            else dist.HalfNormal(1.0),
        )
        if p == 1:
            # HalfNormal gives scalar; wrap as (1,1) matrix
            G = jnp.array([[L_G]])
        else:
            # L_G is (p, p) lower-triangular unit Cholesky from LKJCholesky;
            # scale each row by a positive diagonal
            g_scale = numpyro.sample(
                "g_scale", dist.HalfNormal(jnp.ones(p))
            )
            G = L_G * g_scale[:, None]
    else:
        G = jnp.zeros((p, p))

    # ------------------------------------------------------------------ #
    # Continuous-time mean offset b (CINT)
    # ------------------------------------------------------------------ #
    if config.free_cint:
        b = numpyro.sample("b", dist.Normal(jnp.zeros(p), jnp.ones(p)))
    else:
        b = jnp.zeros(p)

    # ------------------------------------------------------------------ #
    # Observation noise R (diagonal)
    # ------------------------------------------------------------------ #
    log_sigma_obs = numpyro.sample(
        "log_sigma_obs", dist.Normal(jnp.zeros(p), jnp.ones(p))
    )
    R = jnp.diag(jnp.exp(log_sigma_obs) ** 2)

    # ------------------------------------------------------------------ #
    # Initial state mean (per individual via TI predictors)
    # ------------------------------------------------------------------ #
    mu0 = numpyro.sample("mu0", dist.Normal(jnp.zeros(p), 5.0 * jnp.ones(p)))

    if config.n_tipred > 0 and ti_pred is not None:
        T0 = numpyro.sample(
            "T0",
            dist.Normal(
                jnp.zeros((p, config.n_tipred)),
                jnp.ones((p, config.n_tipred)),
            ),
        )
        # m0_i = mu0 + T0 @ ti_pred_i, shape (N, p)
        m0_batch = mu0[None, :] + (T0 @ ti_pred.T).T
    else:
        m0_batch = jnp.broadcast_to(mu0[None, :], (N, p))

    # ------------------------------------------------------------------ #
    # Initial state covariance P0 = L_P0 L_P0^T
    # ------------------------------------------------------------------ #
    L_P0 = numpyro.sample(
        "L_P0",
        dist.LKJCholesky(p, concentration=1.0) if p > 1 else dist.HalfNormal(1.0),
    )
    if p == 1:
        P0 = jnp.array([[L_P0 ** 2]])
    else:
        p0_scale = numpyro.sample("p0_scale", dist.HalfNormal(jnp.ones(p)))
        L_P0_scaled = L_P0 * p0_scale[:, None]
        P0 = L_P0_scaled @ L_P0_scaled.T

    # ------------------------------------------------------------------ #
    # Register computed matrices as deterministic sites for easy retrieval
    # ------------------------------------------------------------------ #
    numpyro.deterministic("A_mat", A)
    numpyro.deterministic("G_mat", G)
    numpyro.deterministic("b_vec", b)
    numpyro.deterministic("R_mat", R)
    numpyro.deterministic("P0_mat", P0)

    # ------------------------------------------------------------------ #
    # Log-likelihood via batched Kalman filter
    # ------------------------------------------------------------------ #
    total_ll, _, _ = batched_filter(
        A, G, b, R, m0_batch, P0, Y, dt_batch, mask_batch,
        free_drift=config.free_drift,
        free_diffusion=config.free_diffusion,
    )
    numpyro.factor("obs", total_ll)
