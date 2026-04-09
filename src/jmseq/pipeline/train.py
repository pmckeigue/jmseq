"""
Pipeline training steps:
  fit_lgssm_fold   — fit LGSSM MAP to longitudinal training data
  run_kalman_fold  — run Kalman filter and return filtered state DataFrame
  fit_poisson_fold — assemble design matrix and fit Poisson GLM MAP
"""

import functools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jmseq.models.lgssm import lgssm_model
from jmseq.models.model_config import LGSSMConfig
from jmseq.models.poisson_glm import poisson_glm_model, qr_decompose
from jmseq.inference.map_estimator import fit_map
from jmseq.kalman.filter import batched_filter
from jmseq.data.transforms import long_to_arrays


@dataclass
class LGSSMResult:
    config: LGSSMConfig
    params: dict          # MAP parameter estimates (JAX arrays, includes deterministics)
    biomarker_cols: list  # names of biomarker / state columns
    id_col: str
    time_col: str


@dataclass
class PoissonResult:
    beta0: float
    beta: np.ndarray      # (P,) coefficients for X columns (no intercept)
    feature_names: list   # ordered list of feature names in X (no intercept)
    Q_ast: np.ndarray     # (N_train, P) QR-rotated design matrix (for reference)
    R_ast_inv: np.ndarray # (P, P) for applying to new X: X_new @ R_ast_inv.T


# ---------------------------------------------------------------------------
# fit_lgssm_fold
# ---------------------------------------------------------------------------

def fit_lgssm_fold(
    long_df: pd.DataFrame,
    config: LGSSMConfig,
    id_col: str = "id",
    time_col: str = "Time",
    biomarker_cols: list | None = None,
    n_steps: int = 3_000,
    lr: float = 0.02,
) -> LGSSMResult:
    """Fit LGSSM MAP to longitudinal training data.

    Parameters
    ----------
    long_df       : output of split_surv_long / trainsplit_long;
                    columns include [id, Time, *biomarkers].
    config        : LGSSMConfig specifying which parameters are free.
    id_col        : individual ID column name.
    time_col      : observation time column name.
    biomarker_cols: list of biomarker column names; inferred from config.p if None.
    n_steps       : SVI steps for MAP estimation.
    lr            : ClippedAdam learning rate.

    Returns
    -------
    LGSSMResult with MAP params and metadata.
    """
    if biomarker_cols is None:
        raise ValueError("biomarker_cols must be specified")

    Y, dt_batch, mask_batch, ids, _ = long_to_arrays(
        long_df, id_col, time_col, biomarker_cols
    )

    model_fn = functools.partial(lgssm_model, config)
    params, losses = fit_map(
        model_fn,
        model_args=(Y, dt_batch, mask_batch),
        n_steps=n_steps,
        lr=lr,
    )
    return LGSSMResult(
        config=config,
        params=params,
        biomarker_cols=biomarker_cols,
        id_col=id_col,
        time_col=time_col,
    )


# ---------------------------------------------------------------------------
# run_kalman_fold
# ---------------------------------------------------------------------------

def run_kalman_fold(
    long_df: pd.DataFrame,
    lgssm_result: LGSSMResult,
) -> pd.DataFrame:
    """Run Kalman filter and return filtered state means as a wide DataFrame.

    Port of R kalmanwide().

    Parameters
    ----------
    long_df      : longitudinal data including test-fold individuals
                   (biomarkers set to NaN post-landmark for test fold).
    lgssm_result : output of fit_lgssm_fold.

    Returns
    -------
    DataFrame with columns [id, tstart, *biomarker_cols] where biomarker columns
    contain the Kalman-filtered state means (posterior mean after each update).
    """
    r = lgssm_result
    Y, dt_batch, mask_batch, ids, times_list = long_to_arrays(
        long_df, r.id_col, r.time_col, r.biomarker_cols
    )

    # Extract computed matrices from MAP params (registered as deterministic sites)
    A   = jnp.array(r.params["A_mat"])
    G   = jnp.array(r.params["G_mat"])
    b   = jnp.array(r.params["b_vec"])
    R   = jnp.array(r.params["R_mat"])
    P0  = jnp.array(r.params["P0_mat"])
    mu0 = jnp.array(r.params["mu0"])

    N = len(ids)
    m0_batch = jnp.broadcast_to(mu0[None, :], (N, r.config.p))

    _, means, _ = batched_filter(
        A, G, b, R, m0_batch, P0, Y, dt_batch, mask_batch,
        free_drift=r.config.free_drift,
        free_diffusion=r.config.free_diffusion,
    )
    means_np = np.array(means)  # (N, max_T, p)

    rows = []
    for i, (id_val, times) in enumerate(zip(ids, times_list)):
        T_i = len(times)
        for t in range(T_i):
            row = {r.id_col: id_val, "tstart": float(times[t])}
            for j, col in enumerate(r.biomarker_cols):
                row[col] = float(means_np[i, t, j])
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fit_poisson_fold
# ---------------------------------------------------------------------------

def _build_poisson_X(
    dataSurv: pd.DataFrame,
    kalman_df: pd.DataFrame,
    timeinvar_surv: list[str],
    biomarker_cols: list[str],
    id_col: str = "id",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Join Kalman states with survival data and assemble design matrix.

    Returns (merged_df, X_no_intercept, log_tobs, feature_names).
    """
    # Join kalman_df (id, tstart, *biomarkers) with dataSurv (id, Time, tstop, event, *timeinvar)
    merged = kalman_df.merge(
        dataSurv[[id_col, "Time", "tstop", "event"] + timeinvar_surv],
        left_on=[id_col, "tstart"],
        right_on=[id_col, "Time"],
        how="inner",
    )
    merged = merged.dropna(subset=biomarker_cols)  # drop rows with missing state estimates
    merged["tobs"] = merged["tstop"] - merged["tstart"]
    merged = merged[merged["tobs"] > 0].reset_index(drop=True)

    feature_names = ["tstart"] + timeinvar_surv + biomarker_cols
    X = merged[feature_names].to_numpy(dtype=np.float32)
    log_tobs = np.log(merged["tobs"].to_numpy(dtype=np.float32))
    return merged, X, log_tobs, feature_names


def fit_poisson_fold(
    dataSurv: pd.DataFrame,
    kalman_df: pd.DataFrame,
    timeinvar_surv: list[str],
    biomarker_cols: list[str],
    id_col: str = "id",
    n_steps: int = 5_000,
    lr: float = 0.05,
) -> PoissonResult:
    """Join Kalman states with survival data and fit Poisson GLM MAP.

    Port of R fit.poissontsplit().

    Parameters
    ----------
    dataSurv      : interval-split survival data with [id, Time, tstop, event, *timeinvar].
    kalman_df     : output of run_kalman_fold; [id, tstart, *biomarker_cols].
    timeinvar_surv: time-invariant covariate names in dataSurv.
    biomarker_cols: biomarker / state column names.
    id_col        : individual ID column name.
    n_steps       : SVI steps.
    lr            : learning rate.

    Returns
    -------
    PoissonResult with MAP estimates of beta0 and beta.
    """
    merged, X_np, log_tobs, feature_names = _build_poisson_X(
        dataSurv, kalman_df, timeinvar_surv, biomarker_cols, id_col
    )
    y = merged["event"].to_numpy(dtype=np.int32)

    X_jax      = jnp.array(X_np)
    y_jax      = jnp.array(y)
    log_tobs_j = jnp.array(log_tobs)

    Q_ast, _, R_ast_inv = qr_decompose(X_jax)

    params, _ = fit_map(
        poisson_glm_model,
        model_args=(Q_ast, y_jax, log_tobs_j),
        model_kwargs={"R_ast_inv": R_ast_inv},
        n_steps=n_steps,
        lr=lr,
    )

    return PoissonResult(
        beta0=float(params["beta0"]),
        beta=np.array(params["beta"]),
        feature_names=feature_names,
        Q_ast=np.array(Q_ast),
        R_ast_inv=np.array(R_ast_inv),
    )
