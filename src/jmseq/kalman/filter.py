"""
Linear Gaussian Kalman filter using jax.lax.scan.

State space model (continuous-time SDE discretised over intervals dt):
  x_{t+1} = A_d(dt) x_t + b_d(dt) + w_t,   w_t ~ N(0, Q_d(dt))
  y_t     = x_t + v_t,                       v_t ~ N(0, R)

Key design: discrete transition matrices (A_d, Q_d, b_d) are precomputed for
all time steps via jax.vmap BEFORE lax.scan.  This keeps matrix exponential
calls out of the scan body, reducing XLA compilation time.

For model variants with fixed A=0 and/or G=0, the static flags `free_drift`
and `free_diffusion` bypass the expm computation entirely using Python-level
conditionals — these are evaluated at JIT-trace time, not at runtime.

Missing observations: obs_mask bool gates updates and log-likelihood
contributions (jnp.where — no branching inside scan).
"""

import jax
import jax.numpy as jnp

from jmseq.kalman.covariance import van_loan, discrete_drift


# ---------------------------------------------------------------------------
# Scan-body factory (closes over R)
# ---------------------------------------------------------------------------

def _make_kalman_step(R):
    p = R.shape[0]
    I = jnp.eye(p)

    def step(carry, inputs):
        m, P, ll = carry
        y, A_d, Q_d, b_d, obs_mask = inputs

        m_pred = A_d @ m + b_d
        P_pred = A_d @ P @ A_d.T + Q_d
        P_pred = 0.5 * (P_pred + P_pred.T)

        innov = y - m_pred
        S     = P_pred + R
        K     = jnp.linalg.solve(S.T, P_pred.T).T
        m_upd = m_pred + K @ innov
        IK    = I - K
        P_upd = IK @ P_pred @ IK.T + K @ R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        m_out = jnp.where(obs_mask, m_upd, m_pred)
        P_out = jnp.where(obs_mask, P_upd, P_pred)

        sign, log_det_S = jnp.linalg.slogdet(S)
        mahal  = innov @ jnp.linalg.solve(S, innov)
        ll_stp = -0.5 * (p * jnp.log(2.0 * jnp.pi) + log_det_S + mahal)
        ll     = ll + jnp.where(obs_mask, ll_stp, 0.0)

        return (m_out, P_out, ll), (m_out, P_out)

    return step


# ---------------------------------------------------------------------------
# Precompute discrete matrices outside the scan
# (static flags bypass expm for fixed-zero A or G)
# ---------------------------------------------------------------------------

def _precompute_transitions(A, G, b, dt_seq, free_drift: bool, free_diffusion: bool):
    """Return (A_d_seq, Q_d_seq, b_d_seq) of shapes (T, p, p), (T, p, p), (T, p).

    When both free_drift=False and free_diffusion=False (model_lmm):
      A_d = I, Q_d = 0 for all t — no expm computation at all.
    When only free_drift=False (model_lmmdiff):
      A_d = I, Q_d precomputed from G only.
    Otherwise full van_loan is called.

    free_drift and free_diffusion are Python bools — evaluated at trace time,
    not runtime — so JAX never includes the expm path in the XLA graph for
    model variants that don't need it.
    """
    T = dt_seq.shape[0]
    p = A.shape[0]

    if not free_drift and not free_diffusion:
        # model_lmm: trivial case — no expm whatsoever
        A_d_seq = jnp.broadcast_to(jnp.eye(p, dtype=A.dtype)[None], (T, p, p))
        Q_d_seq = jnp.zeros((T, p, p), dtype=A.dtype)
        b_d_seq = jnp.outer(dt_seq, b)          # (T, p) = dt * b
        return A_d_seq, Q_d_seq, b_d_seq

    # General case: call van_loan and discrete_drift via vmap over dt
    A_d_seq, Q_d_seq = jax.vmap(lambda dt: van_loan(A, G, dt))(dt_seq)
    b_d_seq          = jax.vmap(lambda dt: discrete_drift(A, b, dt))(dt_seq)
    return A_d_seq, Q_d_seq, b_d_seq


# ---------------------------------------------------------------------------
# Single individual
# ---------------------------------------------------------------------------

def run_filter(A, G, b, R, m0, P0, Y, dt_seq, obs_mask,
               free_drift: bool = True, free_diffusion: bool = True):
    """Run Kalman filter for one individual.

    Parameters
    ----------
    A, G, b, R   : model parameters.
    m0           : (p,) initial state mean.
    P0           : (p, p) initial state covariance.
    Y            : (T, p) observations (0.0 where masked).
    dt_seq       : (T,) time intervals.
    obs_mask     : (T,) bool observation validity.
    free_drift   : Python bool — True if A is a free parameter (not fixed at 0).
    free_diffusion: Python bool — True if G is a free parameter (not fixed at 0).

    Returns
    -------
    ll    : scalar total log-likelihood.
    means : (T, p) filtered means.
    covs  : (T, p, p) filtered covariances.
    """
    A_d_seq, Q_d_seq, b_d_seq = _precompute_transitions(
        A, G, b, dt_seq, free_drift, free_diffusion
    )
    step_fn = _make_kalman_step(R)
    (_, _, ll), (means, covs) = jax.lax.scan(
        step_fn,
        init=(m0, P0, jnp.zeros(())),
        xs=(Y, A_d_seq, Q_d_seq, b_d_seq, obs_mask),
    )
    return ll, means, covs


# ---------------------------------------------------------------------------
# Batched over individuals (vmap)
# ---------------------------------------------------------------------------

def batched_filter(A, G, b, R, m0_batch, P0, Y_batch, dt_batch, mask_batch,
                   free_drift: bool = True, free_diffusion: bool = True):
    """Kalman filter vmapped over the first (individual) axis.

    Parameters
    ----------
    A, G, b, R   : model parameters (shared across individuals).
    m0_batch     : (N, p) per-individual initial means.
    P0           : (p, p) shared initial covariance.
    Y_batch      : (N, T, p) padded observations.
    dt_batch     : (N, T) padded time intervals.
    mask_batch   : (N, T) padded observation masks.
    free_drift, free_diffusion: Python bools passed to _precompute_transitions.

    Returns
    -------
    ll_total : scalar.
    means    : (N, T, p).
    covs     : (N, T, p, p).
    """
    N = m0_batch.shape[0]
    P0_batch = jnp.broadcast_to(P0[None], (N,) + P0.shape)

    _filter_single = jax.vmap(
        lambda m0, P0_i, Y_i, dt_i, mask_i: run_filter(
            A, G, b, R, m0, P0_i, Y_i, dt_i, mask_i, free_drift, free_diffusion
        ),
    )
    lls, means, covs = _filter_single(m0_batch, P0_batch, Y_batch, dt_batch, mask_batch)
    return lls.sum(), means, covs
