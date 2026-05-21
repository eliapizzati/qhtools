"""
Collection of miscellaneous utility routines.
"""

import numpy as np
from numba import jit
from scipy.integrate import cumulative_trapezoid


# Solar luminosity in erg/s
L_SUN_ERG = 3.826e33


def to_ergs(log_L):
    """Convert log-luminosity from solar units to erg/s."""
    return log_L + np.log10(L_SUN_ERG)


def to_solar(log_L):
    """Convert log-luminosity from erg/s to solar units."""
    return log_L - np.log10(L_SUN_ERG)


def get_pimax_from_vmax(vmax, redshift, cosmo=None):
    """Convert a maximum velocity offset to a line-of-sight distance (pi_max).

    Parameters
    ----------
    vmax : float
        Maximum physical velocity offset in km/s.
    redshift : float
        Redshift of the sample.
    cosmo : colossus cosmology instance, optional
        Cosmology to use.  Defaults to the package-wide cosmology.

    Returns
    -------
    float
        Line-of-sight distance pi_max in comoving Mpc (no h-factor).
    """
    if cosmo is None:
        # Lazy import: keeps colossus off the import path for callers that
        # only need the lightweight helpers (to_ergs, get_percentiles, ...).
        from qhtools.utils.cosmology import cosmo
    return vmax * (1.0 + redshift) / cosmo.Hz(redshift)


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


@jit(nopython=True)
def log_mstar_behroozi_18(log_m, z):
    """Behroozi+18 (UniverseMachine) halo mass -> stellar mass (jit-compatible).

    Parameters
    ----------
    log_m : float or ndarray
        log10 halo mass in solar masses.
    z : float
        Redshift.

    Returns
    -------
    log_mstar : float or ndarray
        log10 stellar mass in solar masses.
    """
    eff_0 = -1.431495e+00
    eff_0_a = 1.757030e+00
    eff_0_a2 = 1.350451e+00
    eff_0_z = -2.178460e-01
    m_1 = 1.207402e+01
    m_1_a = 4.599896e+00
    m_1_a2 = 4.423389e+00
    m_1_z = -7.324986e-01
    alpha = 1.973839e+00
    alpha_a = -2.468417e+00
    alpha_a2 = -1.816299e+00
    alpha_z = 1.820800e-01
    beta = 4.702271e-01
    beta_a = -8.751643e-01
    beta_z = -4.866420e-01
    delta = 3.822958e-01
    gamma = -1.160189e+00
    gamma_a = -3.633671e+00
    gamma_z = -1.218900e+00

    a = 1.0 / (1.0 + z)
    a1 = a - 1.0
    lna = np.log(a)

    m_1_f = m_1 + a1 * m_1_a - lna * m_1_a2 + z * m_1_z
    sm_0 = m_1_f + eff_0 + a1 * eff_0_a - lna * eff_0_a2 + z * eff_0_z
    alpha_f = alpha + a1 * alpha_a - lna * alpha_a2 + z * alpha_z
    beta_f = beta + a1 * beta_a + z * beta_z
    delta_f = delta
    gamma_f = 10 ** (gamma + a1 * gamma_a + z * gamma_z)

    dm = log_m - m_1_f
    dm2 = dm / delta_f
    logmstar = (
        sm_0
        - np.log10(10 ** (-alpha_f * dm) + 10 ** (-beta_f * dm))
        + gamma_f * np.exp(-0.5 * (dm2 * dm2))
    )

    return logmstar
