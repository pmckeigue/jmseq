"""
Van Loan method for discrete process noise covariance Q_d(dt).

Given continuous-time SDE:  dX = (A X + b) dt + G dW
the discrete-time transition over interval dt is:
  x_{t+1} = A_d x_t + b_d + w_t,   w_t ~ N(0, Q_d)

where A_d = expm(A * dt) and Q_d is computed via the Van Loan auxiliary matrix.

Van Loan (1978) — avoids matrix inverse, stable near A = 0:
  M = dt * [[-A,  G G^T],
             [ 0,   A^T]]   (shape 2p x 2p)
  expm(M) =: [[...  , F^{-T} Q_d],
              [  0  ,     F^T   ]]
  F   = expm(M)[p:, p:]^T  (== expm(A dt))
  Q_d = F @ expm(M)[:p, p:]

Matrix exponential: we use a fixed-order degree-7 Padé approximant with
3 levels of scaling-squaring.  Unlike jax.scipy.linalg.expm, this contains
NO conditionals (no jax.lax.cond calls), which compiles 10-100× faster when
nested inside jax.vmap / jax.lax.scan.

Accuracy: for ||M/8||_F < 0.5 (satisfied for dt ≤ 10 years and typical LGSSM
parameters) the relative error is < 10^{-10}.
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Fast matrix exponential (no conditionals)
# ---------------------------------------------------------------------------

# Degree-7 Padé coefficients (numerator / denominator coefficients)
_PADE7_C = jnp.array([
    17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.
])


def _expm_pade7(M: jnp.ndarray) -> jnp.ndarray:
    """Degree-7 Padé approximant — valid for ||M||_1 ≲ 1.5, no conditionals."""
    n = M.shape[0]
    I = jnp.eye(n, dtype=M.dtype)
    c = _PADE7_C
    M2 = M @ M
    M4 = M2 @ M2
    M6 = M2 @ M4
    U = M @ (c[7] * M6 + c[5] * M4 + c[3] * M2 + c[1] * I)
    V =     c[6] * M6 + c[4] * M4 + c[2] * M2 + c[0] * I
    # expm(M) = (V - U)^{-1} (V + U)
    return jnp.linalg.solve(V - U, V + U)


def _expm(M: jnp.ndarray) -> jnp.ndarray:
    """Matrix exponential via degree-7 Padé + 3 squarings.

    Scale M by 1/8, apply Padé, then square 3 times.
    Effective input magnitude ≤ ||M||/8; Padé error < 10^{-10} for ||M||/8 ≤ 0.5
    (i.e. ||M|| ≤ 4, which covers dt ≤ 10 yr for typical LGSSM A magnitudes).
    """
    E = _expm_pade7(M / 8.0)
    E = E @ E      # expm(M/4)
    E = E @ E      # expm(M/2)
    E = E @ E      # expm(M)
    return E


# ---------------------------------------------------------------------------
# Van Loan method
# ---------------------------------------------------------------------------

def van_loan(A: jnp.ndarray, G: jnp.ndarray, dt: float | jnp.ndarray):
    """Compute A_d = expm(A dt) and Q_d via the Van Loan method.

    Parameters
    ----------
    A  : (p, p)  drift matrix.
    G  : (p, p)  diffusion matrix (only G G^T enters the formula).
    dt : scalar  time interval (years).

    Returns
    -------
    A_d : (p, p)  discrete transition matrix expm(A dt).
    Q_d : (p, p)  discrete process noise covariance.
    """
    p = A.shape[0]
    GGT = G @ G.T

    top = jnp.concatenate([-A, GGT], axis=1)
    bot = jnp.concatenate([jnp.zeros_like(A), A.T], axis=1)
    M = dt * jnp.concatenate([top, bot], axis=0)   # (2p, 2p)

    eM = _expm(M)

    F   = eM[p:, p:].T          # expm(A dt)
    Q_d = F @ eM[:p, p:]
    Q_d = 0.5 * (Q_d + Q_d.T)  # symmetrise

    return F, Q_d


def discrete_drift(A: jnp.ndarray, b: jnp.ndarray, dt: float | jnp.ndarray):
    """Discrete-time mean increment b_d via augmented matrix exponential.

    x_{t+1} = A_d x_t + b_d + w_t

    Uses A_aug = [[A, b], [0, 0]] so that expm(A_aug dt)[:p, p] = b_d.

    Parameters
    ----------
    A  : (p, p)
    b  : (p,)
    dt : scalar

    Returns
    -------
    b_d : (p,)  discrete mean increment.
    """
    p = A.shape[0]
    A_aug = jnp.block([
        [A,                    b[:, None]],
        [jnp.zeros((1, p)),    jnp.zeros((1, 1))],
    ])
    eM = _expm(dt * A_aug)
    return eM[:p, p]
