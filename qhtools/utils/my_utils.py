"""
Collection of miscellaneous utility routines.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid


# Solar luminosity in erg/s
L_SUN_ERG = 3.826e33


def to_ergs(log_L):
    """Convert log-luminosity from solar units to erg/s."""
    return log_L + np.log10(L_SUN_ERG)


def to_solar(log_L):
    """Convert log-luminosity from erg/s to solar units."""
    return log_L - np.log10(L_SUN_ERG)




def get_percentiles(y, x, percentiles=(0.16, 0.5, 0.84)):
    """
    Compute percentiles of a distribution defined by y(x).

    Integrates y(x) via the trapezoidal rule to build the CDF,
    normalises it, then linearly interpolates to the requested
    percentile levels.

    Parameters
    ----------
    y : array-like
        Function values (e.g. a PDF or unnormalised density).
    x : array-like
        Abscissa values, must be monotonically increasing.
    percentiles : float or array-like, optional
        Fractional percentile(s) in [0, 1].  Default is (0.16, 0.5, 0.84).

    Returns
    -------
    ndarray or float
        The x value(s) at the requested percentile(s).  Returns a scalar
        when *percentiles* is a scalar.
    """
    cdf = cumulative_trapezoid(y, x, initial=0)
    cdf /= cdf[-1]
    return np.interp(percentiles, cdf, x)




def get_median(y, x):
    """
    Compute the median of a distribution defined by y(x).

    Integrates y(x) via the trapezoidal rule to build the CDF, then
    linearly interpolates to find the x value at which the CDF reaches 0.5.

    Parameters
    ----------
    y : array-like
        Function values (e.g. a PDF or unnormalised density).
    x : array-like
        Abscissa values, must be monotonically increasing.

    Returns
    -------
    float
        The x value at which the cumulative integral reaches 50 %.
    """
    return get_percentiles(y, x, 0.5)
