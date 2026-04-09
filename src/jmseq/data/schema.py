"""
Column-name contracts for jmseq DataFrames.

These are the canonical names used throughout the pipeline. User data must
either use these names or pass explicit column-name arguments to each function.
"""

# ---------------------------------------------------------------------------
# Survival data (one row per individual)
# ---------------------------------------------------------------------------
ID_COL       = "id"           # individual identifier
TIME_COL     = "Time"         # interval start time (= tstart after split)
TSTOP_COL    = "tstop"        # interval end time
EVENT_COL    = "event"        # 0 = censored, 1 = event
TIMECENS_COL = "Time.cens"    # raw censoring/event time (pre-split)

# ---------------------------------------------------------------------------
# Longitudinal data
# ---------------------------------------------------------------------------
TSTART_COL   = "tstart"       # observation time (same as Time after split)

# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------
P_EVENT_COL  = "p_event"      # predicted probability of event in interval
SURVPROB_COL = "survprob"     # cumulative survival probability


def validate_surv(df, timeinvar_surv=None):
    """Raise ValueError if required survival columns are missing.

    Parameters
    ----------
    df             : pd.DataFrame to validate.
    timeinvar_surv : optional list of time-invariant covariate names to check.
    """
    required = [ID_COL, TIME_COL, TSTOP_COL, EVENT_COL]
    if timeinvar_surv:
        required += list(timeinvar_surv)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Survival DataFrame missing columns: {missing}")


def validate_long(df, biomarker_cols):
    """Raise ValueError if required longitudinal columns are missing.

    Parameters
    ----------
    df            : pd.DataFrame to validate.
    biomarker_cols: list of biomarker column names to check.
    """
    required = [ID_COL, TIME_COL] + list(biomarker_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Longitudinal DataFrame missing columns: {missing}")
