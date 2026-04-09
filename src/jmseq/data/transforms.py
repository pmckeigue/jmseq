"""
Data transforms: interval splitting and array conversion.

split_surv_long  — port of R split.SurvLong()
long_to_arrays   — convert longitudinal DataFrame to padded JAX arrays for Kalman filter
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# split_surv_long
# ---------------------------------------------------------------------------

def split_surv_long(
    surv_df: pd.DataFrame,
    long_df: pd.DataFrame,
    max_interval: float,
    timeinvar_surv: list[str],
    biomarkers: list[str],
    id_col: str = "id",
    time_col: str = "Time",
    timecens_col: str = "Time.cens",
    event_col: str = "event",
) -> pd.DataFrame:
    """Split person-time intervals so none is longer than max_interval.

    Port of R split.SurvLong().

    Parameters
    ----------
    surv_df       : one row per individual; columns [id, Time.cens, event, *timeinvar_surv].
    long_df       : one row per observation; columns [id, Time, *biomarkers, ...].
    max_interval  : maximum allowed interval length (e.g. 0.25 years).
    timeinvar_surv: time-invariant covariate names in surv_df.
    biomarkers    : biomarker column names in long_df.
    id_col        : name of the individual ID column (default "id").
    time_col      : name of the observation time column in long_df (default "Time").
    timecens_col  : name of the censoring time column in surv_df (default "Time.cens").
    event_col     : name of the event indicator column (default "event").

    Returns
    -------
    DataFrame with columns [id, Time (=tstart), tstop, event, *timeinvar_surv, *biomarkers].
    """
    surv_keep = [id_col, timecens_col, event_col] + timeinvar_surv
    merged = long_df[[id_col, time_col] + biomarkers].merge(
        surv_df[surv_keep], on=id_col, how="left"
    )
    merged = merged.sort_values([id_col, time_col]).reset_index(drop=True)

    # Compute tstop: next Time within id, or Time.cens for last observation
    merged["tstop"] = merged.groupby(id_col)[time_col].shift(-1)
    is_last = merged["tstop"].isna()
    merged.loc[is_last, "tstop"] = merged.loc[is_last, timecens_col]

    # event = 0 for all non-last observations per individual
    merged.loc[~is_last, event_col] = 0

    # Rename time_col to Time for clarity in output
    merged = merged.rename(columns={time_col: "Time"})

    # Split long intervals row by row
    out_rows = []
    for _, row in merged.iterrows():
        row_dict = row.to_dict()
        tstart = row_dict["Time"]
        tstop  = row_dict["tstop"]
        interval = tstop - tstart

        if interval <= max_interval + 1e-9:
            out_rows.append(row_dict)
        else:
            # Generate split points at tstart + k * max_interval
            n_cuts = int(np.floor(interval / max_interval))
            cut_pts = [tstart + max_interval * k for k in range(1, n_cuts + 1)]
            cut_pts = [c for c in cut_pts if c < tstop - 1e-9]

            boundaries = [tstart] + cut_pts + [tstop]
            n_sub = len(boundaries) - 1
            orig_event = row_dict[event_col]

            for k in range(n_sub):
                new = row_dict.copy()
                new["Time"]  = boundaries[k]
                new["tstop"] = boundaries[k + 1]
                # Only the last sub-interval inherits the original event
                new[event_col] = orig_event if k == n_sub - 1 else 0
                # Inserted sub-intervals (k > 0) have no biomarker observation
                if k > 0:
                    for bio in biomarkers:
                        new[bio] = np.nan
                out_rows.append(new)

    result = pd.DataFrame(out_rows)
    result = result.drop(columns=[timecens_col], errors="ignore")
    # Reorder columns
    col_order = [id_col, "Time", "tstop", event_col] + timeinvar_surv + biomarkers
    col_order = [c for c in col_order if c in result.columns]
    return result[col_order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# long_to_arrays
# ---------------------------------------------------------------------------

def long_to_arrays(
    long_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    biomarker_cols: list[str],
    max_T: int | None = None,
) -> tuple:
    """Convert longitudinal DataFrame to padded JAX arrays for the Kalman filter.

    Time step for step 0 is Time[0] (assuming state origin at t=0).
    Time step for step t>0 is Time[t] - Time[t-1].
    Missing observations (any biomarker NaN) have obs_mask=False.

    Parameters
    ----------
    long_df       : DataFrame sorted by id, time; one row per person-time interval.
    id_col        : column name for individual ID.
    time_col      : column name for interval start time.
    biomarker_cols: column names of biomarkers (observation variables).
    max_T         : pad length; defaults to max sequence length in data.

    Returns
    -------
    Y          : (N, max_T, p)  float32 JAX array (0.0 where masked).
    dt_batch   : (N, max_T)     float32 JAX array (0.0 for padding steps).
    mask_batch : (N, max_T)     bool JAX array (False for padding steps).
    ids        : list of N individual IDs (in order).
    times_list : list of N numpy arrays, each containing the actual Time values.
    """
    p = len(biomarker_cols)
    df = long_df.sort_values([id_col, time_col])
    groups = list(df.groupby(id_col, sort=False))
    # Restore original id order
    id_order = df[id_col].drop_duplicates().tolist()
    group_dict = {gid: gdf for gid, gdf in groups}

    N = len(id_order)
    if max_T is None:
        max_T = max(len(group_dict[gid]) for gid in id_order)

    Y_np        = np.zeros((N, max_T, p), dtype=np.float32)
    dt_np       = np.zeros((N, max_T),    dtype=np.float32)
    mask_np     = np.zeros((N, max_T),    dtype=bool)
    times_list  = []
    ids         = []

    for i, gid in enumerate(id_order):
        gdf = group_dict[gid].reset_index(drop=True)
        T_i = len(gdf)
        times_f64 = gdf[time_col].to_numpy(dtype=np.float64)  # original precision
        times_f32 = times_f64.astype(np.float32)
        bio_vals = gdf[biomarker_cols].to_numpy(dtype=np.float32)  # (T_i, p)

        # dt: first step from 0, subsequent steps are differences (float64 for accuracy)
        dt_i = np.empty(T_i, dtype=np.float32)
        dt_i[0] = times_f64[0]
        if T_i > 1:
            dt_i[1:] = np.diff(times_f64).astype(np.float32)
        # Clamp to zero (dt=0 is valid: expm(0)=I, Q_d=0, no-op predict step)
        dt_i = np.maximum(dt_i, 0.0)

        # obs_mask: True if at least one biomarker is observed (not all NaN)
        obs_i = ~np.all(np.isnan(bio_vals), axis=1)

        Y_np[i, :T_i, :]    = np.where(np.isnan(bio_vals), 0.0, bio_vals)
        dt_np[i, :T_i]      = dt_i
        mask_np[i, :T_i]    = obs_i

        times_list.append(times_f64)  # float64 original values for DataFrame joins
        ids.append(gid)

    return (
        jnp.array(Y_np),
        jnp.array(dt_np),
        jnp.array(mask_np),
        ids,
        times_list,
    )
