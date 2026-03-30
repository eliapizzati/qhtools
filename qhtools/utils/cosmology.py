"""Cosmology configuration using Colossus.

This module provides a single, shared cosmology instance for the package.
By default the FLAMINGO (3x2pt + all) parameters are used, but the active
cosmology can be switched at any time via :func:`set_cosmology`.

Usage
-----
>>> from qhtools.utils.cosmology import cosmo
>>> cosmo.luminosityDistance(2.0)   # uses the currently active cosmology

To switch cosmology::

    from qhtools.utils.cosmology import set_cosmology

    # Use a built-in Colossus cosmology
    set_cosmology("planck18")

    # Or supply custom parameters
    set_cosmology("my_cosmo", params={"flat": True, "H0": 70.0, ...})

    # Reset back to the default
    set_cosmology()
"""

import numpy as np
from colossus.cosmology import cosmology as _colossus_cosmology

# ---------------------------------------------------------------------------
# FLAMINGO parameters (default)
# ---------------------------------------------------------------------------

_DEFAULT_NAME = "Flamingo_Cosmo"


def get_flamingo_params():
    """Return the FLAMINGO cosmological parameters.

    The values correspond to the *3x2pt + all* constraints used in the
    FLAMINGO simulation suite (non-flat w0-wa dark-energy model).

    Returns
    -------
    dict
        Parameter dictionary accepted by ``colossus.cosmology.addCosmology``.
    """
    return {
        "flat": False,
        "H0": 68.10015470019941,
        "Om0": 0.304611,
        "Ob0": 0.0486,
        "Ode0": 0.693922,
        "sigma8": 0.80765,
        "ns": 0.9667,
        "de_model": "w0wa",
        "w0": -1.0,
        "wa": 0.0,
        "Tcmb0": 2.7255,
        "Neff": 3.04400163,
    }


def set_cosmology(name=None, params=None):
    """Set the active cosmology for the package.

    Parameters
    ----------
    name : str, optional
        Name of the cosmology. If *params* is ``None`` this must be a
        cosmology already known to Colossus (e.g. ``"planck18"``).
        When omitted (or ``None``), the default FLAMINGO cosmology is used.
    params : dict, optional
        Custom parameter dictionary (same format as
        ``colossus.cosmology.addCosmology``). When provided, the cosmology
        is registered under *name* before being activated.

    Returns
    -------
    colossus.cosmology.Cosmology
        The newly active cosmology instance.
    """
    global cosmo

    if name is None and params is None:
        # Reset to default FLAMINGO cosmology
        name = _DEFAULT_NAME
        params = get_flamingo_params()

    if params is not None:
        if name is None:
            raise ValueError("A name is required when supplying custom params")
        _colossus_cosmology.addCosmology(name, params)

    cosmo = _colossus_cosmology.setCosmology(name)
    return cosmo


# Register and activate the default cosmology at import time.
_colossus_cosmology.addCosmology(_DEFAULT_NAME, get_flamingo_params())
cosmo = _colossus_cosmology.setCosmology(_DEFAULT_NAME)


if __name__ == "__main__":
    # Quick diagnostic plot: peak height (nu) vs. halo mass at several redshifts.
    from colossus.lss import peaks
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    logm_axis = np.linspace(9, 16.5, 1000)
    redshifts = np.arange(0, 8)

    for z in redshifts:
        # Colossus expects masses in M_sun/h
        mass_h = 10 ** (logm_axis + np.log10(cosmo.h))
        nu = peaks.peakHeight(mass_h, z)
        ax.plot(logm_axis, nu, label=f"z={z:.0f}")

    ax.set_xlabel(r"$\log_{10}(M\;/\;M_\odot)$")
    ax.set_ylabel(r"$\nu$ (peak height)")
    ax.legend()
    plt.show()

