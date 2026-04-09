"""
Prediction and evaluation on test folds.

predict_testdata    — port of R predict.testdata()
tabulate_predictions — port of R tabulate.predictions()
"""

import numpy as np
import pandas as pd

from jmseq.pipeline.train import PoissonResult


def predict_testdata(
    dataSurv: pd.DataFrame,
    kalman_df: pd.DataFrame,
    poisson_result: PoissonResult,
    timeinvar_surv: list[str],
    biomarker_cols: list[str],
    landmark_time: float,
    id_col: str = "id",
) -> pd.DataFrame:
    """Predict event probability for test-fold individuals post-landmark.

    Port of R predict.testdata().

    Parameters
    ----------
    dataSurv        : full interval-split survival data [id, Time, tstop, event, *timeinvar].
    kalman_df       : Kalman filter output pre-filtered to test subjects and post-landmark
                      intervals: kalman_df[kalman_df[id_col].isin(ids_test) &
                                           (kalman_df["tstart"] > landmark_time)]
    poisson_result  : output of fit_poisson_fold.
    timeinvar_surv  : time-invariant covariate names.
    biomarker_cols  : biomarker / state column names.
    landmark_time   : start of prediction window.
    id_col          : individual ID column name.

    Returns
    -------
    DataFrame with columns [id, tstart, tstop, event, tobs, *covariates, p_event, survprob].
    """
    # Join kalman_df with dataSurv to get tstop, event, and time-invariant covariates
    merged = kalman_df.merge(
        dataSurv[[id_col, "Time", "tstop", "event"] + timeinvar_surv],
        left_on=[id_col, "tstart"],
        right_on=[id_col, "Time"],
        how="inner",
    )

    # Compute Time.cens per individual as max(tstop)
    merged["Time.cens"] = merged.groupby(id_col)["tstop"].transform("max")

    # tobs: use max interval length for last intervals (matching R's tobs.max logic)
    tobs_max = (merged["tstop"] - merged["tstart"]).max()
    last_mask = merged["tstop"] == merged["Time.cens"]
    merged.loc[last_mask, "tstop"] = merged.loc[last_mask, "tstart"] + tobs_max

    merged["tobs"] = merged["tstop"] - merged["tstart"]

    # Recompute Time.cens and set event=0 for non-final intervals
    merged["Time.cens"] = merged.groupby(id_col)["tstop"].transform("max")
    merged.loc[merged["tstop"] < merged["Time.cens"], "event"] = 0

    merged = merged.dropna(subset=biomarker_cols).reset_index(drop=True)

    # Build design matrix (same feature order as training, no intercept)
    r = poisson_result
    X = merged[r.feature_names].to_numpy(dtype=np.float32)
    log_tobs = np.log(merged["tobs"].to_numpy(dtype=np.float32))

    # Predict: log hazard rate per unit time = beta0 + X @ beta
    log_rate = r.beta0 + X @ r.beta
    predicted_events = np.exp(log_tobs) * np.exp(log_rate)   # tobs * rate

    merged["p_event"] = 1.0 - np.exp(-predicted_events)

    # Cumulative survival probability per individual (product of (1 - p_event))
    merged = merged.sort_values([id_col, "tstart"]).reset_index(drop=True)
    merged["survprob"] = (
        merged.groupby(id_col)["p_event"]
        .transform(lambda x: (1.0 - x).cumprod())
    )

    keep_cols = (
        [id_col, "tstart", "tstop", "event", "tobs"]
        + timeinvar_surv
        + biomarker_cols
        + ["p_event", "survprob"]
    )
    keep_cols = [c for c in keep_cols if c in merged.columns]
    return merged[keep_cols].drop(columns=["Time.cens"], errors="ignore")


def tabulate_predictions(pred_df: pd.DataFrame) -> dict:
    """Summary statistics for predictive performance.

    Port of R tabulate.predictions().

    The C-statistic is the ROC AUC over all person-time intervals from the
    landmark time to exit, with p_event as the score and the interval-level
    event indicator as the outcome — computed identically to the log-score.

    Parameters
    ----------
    pred_df : output of predict_testdata; must have columns [event, p_event, tobs].

    Returns
    -------
    Dict with keys: Observed, Predicted, Person-years, Log score, C-statistic.
    """
    from jmseq.utils.metrics import log_score as _log_score, c_statistic as _c_statistic

    df = pred_df.dropna(subset=["event", "p_event", "tobs"])

    observed   = int(df["event"].sum())
    predicted  = float(df["p_event"].sum())
    person_yrs = float(df["tobs"].sum())
    ls         = _log_score(df["event"].to_numpy(), df["p_event"].to_numpy())
    c_stat     = _c_statistic(df["event"].to_numpy(), df["p_event"].to_numpy())

    return {
        "Observed": observed,
        "Predicted": predicted,
        "Person-years": person_yrs,
        "Log score": ls,
        "C-statistic": c_stat,
    }
