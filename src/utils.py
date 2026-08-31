"""Shared helpers: t-based confidence intervals for cross-validated metrics."""
import numpy as np
from scipy import stats


def mean_ci(values, confidence=0.95):
    """Return (mean, half_width) for a t-based CI over a 1D array of fold scores."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return mean, float("nan")
    sem = values.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, t_crit * sem


def fmt_ci(mean, half_width, decimals=3):
    return f"{mean:.{decimals}f} ± {half_width:.{decimals}f}"
