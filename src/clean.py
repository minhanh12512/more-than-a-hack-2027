"""Small cleaning helpers used by src/data.py. Nothing in here reads a file."""

import numpy as np
import pandas as pd


def fill_short_gaps(values, max_gap):
    """Fill runs of missing values no longer than `max_gap` frames with a straight line.

    A half-second hole in the ball track is safe to bridge: the ball went from the last place it
    was seen to the next. Longer holes are left empty, because a guess there would be invented
    data. Gaps at the very start or end are never filled.
    """
    series = pd.Series(values, dtype="float64")
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    short = missing & (run_len <= max_gap)
    bridged = series.interpolate(limit_area="inside")
    series[short] = bridged[short]
    return series.to_numpy()


def speed(x, y, fps):
    """Speed in metres per second between consecutive frames. NaN where either end is missing."""
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    return step * fps


def nearest_distance(px, py, bx, by, columns):
    """Distance in metres from the ball to the closest player among `columns`, per frame."""
    d = np.hypot(px[:, columns] - bx[:, None], py[:, columns] - by[:, None])
    d = np.where(np.isnan(d), np.inf, d).min(axis=1)
    d[np.isinf(d)] = np.nan
    return d
