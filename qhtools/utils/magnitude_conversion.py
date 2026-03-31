"""
Conversions between quasar magnitude systems (M_1450, M_i(z=2),
apparent magnitudes) and bolometric luminosity, plus the associated
Jacobian transformations for luminosity-function number densities.

Supported bolometric-correction methods
----------------------------------------
- **Shen+09** — linear relation via M_i(z=2) intermediate.
- **Runnoe+12** — monochromatic-luminosity relation with slope *B = 0.910*
  and bolometric correction factor 0.75.

Both reduce to a linear relation  log L_bol = C + S * M_1450  whose
constants are precomputed at import time (from astropy unit conversions)
so that the per-call cost is a single multiply-add.
"""

import numpy as np
import astropy.units as u

from qhtools.utils.cosmology import cosmo as _default_cosmo

# ---------------------------------------------------------------------------
# Precomputed constants (Runnoe+12)
# ---------------------------------------------------------------------------
# The Runnoe+12 conversion expands to:
#   log_Lbol = log10(factor) + A + B * log10(nu_1450 * Lnu_1450)
# where Lnu_1450 = 4*pi*(10 pc)^2 * 3631 Jy * 10^(-0.4 M).
# This is linear in M:  log_Lbol = _RUNNOE_CONST + _RUNNOE_SLOPE * M_1450.

_RUNNOE_FACTOR = 0.75
_RUNNOE_A = 4.745
_RUNNOE_B = 0.910

_NU_1450 = (1450.0 * u.angstrom).to("Hz", equivalencies=u.spectral()).value
_C_FLUX = (
    4.0 * np.pi * (10.0 * u.pc) ** 2 * 3631.0 * u.Jansky
).to(u.erg / u.s / u.Hz).value

_RUNNOE_CONST = (
    np.log10(_RUNNOE_FACTOR)
    + _RUNNOE_A
    + _RUNNOE_B * (np.log10(_NU_1450) + np.log10(_C_FLUX))
)
_RUNNOE_SLOPE = -0.4 * _RUNNOE_B  # -0.364

# ---------------------------------------------------------------------------
# Precomputed constants (Shen+09)
# ---------------------------------------------------------------------------
# log_Lbol = 0.4 * (90 - (M_1450 - 1.28)) = 36.512 - 0.4 * M_1450

_SHEN_CONST = 0.4 * (90.0 + 1.28)  # 36.512
_SHEN_SLOPE = -0.4

# ---------------------------------------------------------------------------
# Jacobians: |dM / d(log L)|
# ---------------------------------------------------------------------------
_SHEN_JACOBIAN = abs(1.0 / _SHEN_SLOPE)       # 2.5
_RUNNOE_JACOBIAN = abs(1.0 / _RUNNOE_SLOPE)   # 2.5 / B


# ---------------------------------------------------------------------------
# K-correction table for apparent -> absolute magnitude
# ---------------------------------------------------------------------------
# Piecewise-constant k-corrections by redshift band.
# TODO: replace with a smooth interpolating function covering all redshifts.
_K_CORR_EDGES = np.array([1.0, 1.5, 2.5, 3.5, 4.5, 5.0, 6.0])
_K_CORR_VALUES = np.array([-1.3, -1.6, -1.75, -1.9, -2.0, -2.2])


# ========================================================================== #
#  Apparent / absolute magnitude conversions                                  #
# ========================================================================== #

def get_M1450_from_m(mag, redshift, cosmo=_default_cosmo):
    """
    Convert apparent magnitude to M_1450 absolute magnitude.

    Applies a piecewise k-correction that depends on the redshift band
    (i-band for z < 5, z-band for z >= 5) and the Colossus distance modulus.

    Parameters
    ----------
    mag : float or array-like
        Apparent magnitude.
    redshift : float
        Source redshift (must be in [1, 6]).
    cosmo : colossus Cosmology, optional
        Cosmology instance for the distance modulus.

    Returns
    -------
    float or ndarray
        Absolute magnitude M_1450.
    """
    if redshift < 1.0 or redshift > 6.0:
        raise ValueError("Redshift must be between 1.0 and 6.0")

    idx = np.searchsorted(_K_CORR_EDGES[1:], redshift)
    k_correction = _K_CORR_VALUES[idx]

    return mag - cosmo.distanceModulus(redshift) - k_correction


def get_M1450_from_Mi_z2(M_i_z2):
    """
    Convert M_i(z=2) to M_1450 using the Lusso+15 offset.

    Parameters
    ----------
    M_i_z2 : float or array-like
        Absolute i-band magnitude normalised to z = 2.

    Returns
    -------
    float or ndarray
        M_1450 absolute magnitude.
    """
    return M_i_z2 + 1.28


# ========================================================================== #
#  M_1450  <->  log L_bol                                                     #
# ========================================================================== #

def get_log_Lbol_from_M1450(M_1450, method="Runnoe+12"):
    """
    Convert M_1450 to log10 bolometric luminosity [erg/s].

    Parameters
    ----------
    M_1450 : float or array-like
        Absolute magnitude at 1450 A.
    method : {'Runnoe+12', 'Shen+09'}
        Bolometric correction to apply.

    Returns
    -------
    float or ndarray
        log10(L_bol / [erg/s]).
    """
    if method == "Runnoe+12":
        return _RUNNOE_CONST + _RUNNOE_SLOPE * M_1450
    if method == "Shen+09":
        return _SHEN_CONST + _SHEN_SLOPE * M_1450
    raise ValueError("method must be 'Shen+09' or 'Runnoe+12'")


def get_M1450_from_log_Lbol(log_Lbol, method="Runnoe+12"):
    """
    Convert log10 bolometric luminosity [erg/s] to M_1450.

    Parameters
    ----------
    log_Lbol : float or array-like
        log10(L_bol / [erg/s]).
    method : {'Runnoe+12', 'Shen+09'}
        Bolometric correction to invert.

    Returns
    -------
    float or ndarray
        Absolute magnitude M_1450.
    """
    if method == "Runnoe+12":
        return (log_Lbol - _RUNNOE_CONST) / _RUNNOE_SLOPE
    if method == "Shen+09":
        return (log_Lbol - _SHEN_CONST) / _SHEN_SLOPE
    raise ValueError("method must be 'Shen+09' or 'Runnoe+12'")


# ========================================================================== #
#  QLF Jacobians:  phi per mag  <->  phi per dex                              #
# ========================================================================== #

def get_logphi_dm_from_logphi_dlogL(logphi_dlogL, method="Runnoe+12"):
    """
    Convert log number density per dex to log number density per magnitude.

    Parameters
    ----------
    logphi_dlogL : float or array-like
        log10(phi) in units of dex^-1.
    method : {'Runnoe+12', 'Shen+09'}

    Returns
    -------
    float or ndarray
        log10(phi) in units of mag^-1.
    """
    if method == "Runnoe+12":
        return logphi_dlogL - np.log10(_RUNNOE_JACOBIAN)
    if method == "Shen+09":
        return logphi_dlogL - np.log10(_SHEN_JACOBIAN)
    raise ValueError("method must be 'Shen+09' or 'Runnoe+12'")


def get_logphi_dlogL_from_logphi_dm(logphi_dm, method="Runnoe+12"):
    """
    Convert log number density per magnitude to log number density per dex.

    Parameters
    ----------
    logphi_dm : float or array-like
        log10(phi) in units of mag^-1.
    method : {'Runnoe+12', 'Shen+09'}

    Returns
    -------
    float or ndarray
        log10(phi) in units of dex^-1.
    """
    if method == "Runnoe+12":
        return logphi_dm + np.log10(_RUNNOE_JACOBIAN)
    if method == "Shen+09":
        return logphi_dm + np.log10(_SHEN_JACOBIAN)
    raise ValueError("method must be 'Shen+09' or 'Runnoe+12'")


def get_phi_dlogL_from_phi_dm(phi_dm, method="Runnoe+12"):
    """
    Convert number density per magnitude to number density per dex (linear).

    Parameters
    ----------
    phi_dm : float or array-like
        Number density in units of mag^-1.
    method : {'Runnoe+12', 'Shen+09'}

    Returns
    -------
    float or ndarray
        Number density in units of dex^-1.
    """
    if method == "Runnoe+12":
        return phi_dm * _RUNNOE_JACOBIAN
    if method == "Shen+09":
        return phi_dm * _SHEN_JACOBIAN
    raise ValueError("method must be 'Shen+09' or 'Runnoe+12'")


# ========================================================================== #
#  Quick visual test                                                           #
# ========================================================================== #

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    M_1450 = np.linspace(-26.7, -20.0, 100)
    for method, color in [("Runnoe+12", "k"), ("Shen+09", "r")]:
        axes[0].plot(M_1450, get_log_Lbol_from_M1450(M_1450, method), color=color, lw=2, label=method)
    axes[0].set(xlabel=r"$M_{1450}$", ylabel=r"$\log_{10}(L_{\rm bol})$")
    axes[0].legend()

    log_Lbol = np.linspace(40.0, 48.0, 100)
    for method, color in [("Runnoe+12", "k"), ("Shen+09", "r")]:
        axes[1].plot(log_Lbol, get_M1450_from_log_Lbol(log_Lbol, method), color=color, lw=2, label=method)
    axes[1].set(xlabel=r"$\log_{10}(L_{\rm bol})$", ylabel=r"$M_{1450}$")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
