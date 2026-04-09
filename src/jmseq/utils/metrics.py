"""
Predictive performance metrics for joint models.

Functions
---------
log_score         — per-interval Bernoulli log-probability, summed
c_statistic       — ROC AUC over all person-time intervals (interval-level C)
calibration_table — O/E ratio and expected events by decile of predicted risk
"""

import numpy as np
import pandas as pd


def log_score(events: np.ndarray, p_event: np.ndarray) -> float:
    """Sum of Bernoulli log-probabilities over all intervals.

    Parameters
    ----------
    events  : (N,) 0/1 event indicator per interval.
    p_event : (N,) predicted event probability per interval.

    Returns
    -------
    Scalar log-score (higher is better; 0 is perfect).
    """
    eps = 1e-12
    p = np.clip(p_event, eps, 1.0 - eps)
    return float(np.sum(events * np.log(p) + (1.0 - events) * np.log(1.0 - p)))


def c_statistic(events: np.ndarray, p_event: np.ndarray) -> float:
    """C-statistic computed as ROC AUC over all person-time intervals.

    Each interval from the landmark time to exit contributes one observation,
    with a binary outcome (event / no event) and a predicted probability
    p_event.  The C-statistic is the area under the ROC curve of p_event
    predicting the interval-level event indicator — exactly analogous to how
    the log-score is computed across the same set of intervals.

    Parameters
    ----------
    events  : (N,) 0/1 event indicator per interval.
    p_event : (N,) predicted event probability per interval (higher = higher risk).

    Returns
    -------
    C-statistic in [0.5, 1.0] (0.5 = chance, 1.0 = perfect discrimination).
    """
    from sklearn.metrics import roc_auc_score
    events  = np.asarray(events,  dtype=np.float64)
    p_event = np.asarray(p_event, dtype=np.float64)
    if len(np.unique(events)) < 2:
        return float("nan")
    return float(roc_auc_score(events, p_event))


def calibration_table(
    events: np.ndarray,
    p_event: np.ndarray,
    n_groups: int = 10,
) -> pd.DataFrame:
    """Observed vs expected events by decile of predicted probability.

    Parameters
    ----------
    events  : (N,) 0/1 event indicator per interval.
    p_event : (N,) predicted event probability per interval.
    n_groups: number of quantile groups (default 10 = deciles).

    Returns
    -------
    DataFrame with columns: group, n, observed, expected, O_E_ratio.
    """
    df = pd.DataFrame({"event": events, "p_event": p_event})
    df["group"] = pd.qcut(df["p_event"], q=n_groups, labels=False, duplicates="drop")
    tbl = (
        df.groupby("group")
        .agg(n=("event", "count"), observed=("event", "sum"), expected=("p_event", "sum"))
        .reset_index()
    )
    tbl["O_E_ratio"] = tbl["observed"] / tbl["expected"].clip(lower=1e-9)
    return tbl
