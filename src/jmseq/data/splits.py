"""
Test/train split helpers.

Port of R trainsplit.surv() and trainsplit.long().
"""

import numpy as np
import pandas as pd


def trainsplit_surv(
    ids_test,
    dataSurv: pd.DataFrame,
    landmark_time: float,
    id_col: str = "id",
    tstop_col: str = "tstop",
) -> pd.DataFrame:
    """Remove test-fold intervals that start after landmark_time.

    Port of R trainsplit.surv(): drops rows where id is in ids_test AND
    tstop > landmark_time.

    Parameters
    ----------
    ids_test       : collection of individual IDs in the test fold.
    dataSurv       : DataFrame with columns [id, Time, tstop, event, ...].
    landmark_time  : landmark time; test-fold observations after this are dropped.

    Returns
    -------
    Training survival DataFrame.
    """
    ids_test_set = set(ids_test)
    mask_test_post = (
        dataSurv[id_col].isin(ids_test_set) & (dataSurv[tstop_col] > landmark_time)
    )
    return dataSurv[~mask_test_post].reset_index(drop=True)


def trainsplit_long(
    ids_test,
    dataLong: pd.DataFrame,
    landmark_time: float,
    biomarkers: list[str],
    id_col: str = "id",
    time_col: str = "Time",
) -> pd.DataFrame:
    """Blank out biomarker values after landmark_time for test-fold individuals.

    Port of R trainsplit.long(): sets biomarker columns to NaN where id is in
    ids_test AND Time > landmark_time. All rows are retained (the Kalman filter
    needs to propagate the state forward through the masked steps).

    Parameters
    ----------
    ids_test      : collection of individual IDs in the test fold.
    dataLong      : DataFrame with columns [id, Time, tstop, *biomarkers, ...].
    landmark_time : landmark time.
    biomarkers    : names of biomarker columns to blank out.

    Returns
    -------
    Copy of dataLong with biomarkers set to NaN for test fold post-landmark.
    """
    out = dataLong.copy()
    ids_test_set = set(ids_test)
    mask = out[id_col].isin(ids_test_set) & (out[time_col] > landmark_time)
    out.loc[mask, biomarkers] = np.nan
    return out
