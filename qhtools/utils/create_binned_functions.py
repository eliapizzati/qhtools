"""
Utilities for constructing binned distribution functions
(luminosity functions, mass functions, etc.) from catalogue data.
"""

import numpy as np


def create_luminosity_function(
    quasar_luminosities,
    box_volume,
    input_type="L_bol",
    lowest_lim=1e42,
    highest_lim=1e48,
    n_bins=51,
    minimum_in_bin=3,
):
    """
    Construct a binned quasar luminosity function (QLF) from a luminosity catalogue.

    Bins are linearly spaced for absolute magnitudes (``M_1450``) and
    log-spaced for bolometric luminosities (``L_bol``).  Bins containing
    fewer than *minimum_in_bin* objects are discarded.

    Parameters
    ----------
    quasar_luminosities : array-like
        Luminosity catalogue values.
    box_volume : float
        Comoving volume of the simulation box (consistent units with the
        desired output number density).
    input_type : {'M_1450', 'L_bol'}, optional
        Whether the input luminosities are absolute magnitudes at 1450 A
        or bolometric luminosities.
    lowest_lim : float, optional
        Lower edge of the binning range.
    highest_lim : float, optional
        Upper edge of the binning range.
    n_bins : int, optional
        Number of bins.
    minimum_in_bin : int, optional
        Minimum object count for a bin to be kept.

    Returns
    -------
    bin_centers : ndarray
        Centre of each valid bin.
    qlf : ndarray
        Number density per unit magnitude (or per dex) in each bin.
    error : ndarray
        Poisson uncertainty on the number density.
    """
    quasar_luminosities = np.ascontiguousarray(quasar_luminosities)

    if input_type == "M_1450":
        bins = np.linspace(lowest_lim, highest_lim, n_bins + 1)
        bin_width = bins[1] - bins[0]
        normalization_factor = 1.0 / (bin_width * box_volume)

    elif input_type == "L_bol":
        log_low, log_high = np.log10(lowest_lim), np.log10(highest_lim)
        bins = np.logspace(log_low, log_high, n_bins + 1)
        bin_width_dex = (log_high - log_low) / n_bins
        normalization_factor = 1.0 / (bin_width_dex * box_volume)

    else:
        raise ValueError(
            f"Unrecognised input_type '{input_type}'. Use 'M_1450' or 'L_bol'."
        )

    counts, _ = np.histogram(quasar_luminosities, bins)
    valid = counts >= minimum_in_bin
    counts_valid = counts[valid]

    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    return (
        bin_centers[valid],
        counts_valid * normalization_factor,
        np.sqrt(counts_valid) * normalization_factor,
    )


def create_mass_function(
    masses,
    box_volume,
    lowest_mass=1e8,
    highest_mass=1e16,
    n_bins=45,
    minimum_in_bin=3,
    return_n_counts=False,
):
    """
    Construct a binned halo mass function from a halo mass catalogue.

    Bins are log-spaced between *lowest_mass* and *highest_mass*.
    Bins containing fewer than *minimum_in_bin* haloes are discarded.

    Parameters
    ----------
    masses : array-like
        Halo mass catalogue values.
    box_volume : float
        Comoving volume of the simulation box.
    lowest_mass : float, optional
        Lower mass edge of the binning range.
    highest_mass : float, optional
        Upper mass edge of the binning range.
    n_bins : int, optional
        Number of bins.
    minimum_in_bin : int, optional
        Minimum halo count for a bin to be kept.
    return_n_counts : bool, optional
        If ``True``, also return the raw object counts per valid bin.

    Returns
    -------
    bin_centers : ndarray
        Centre of each valid bin (linear average of bin edges).
    mass_function : ndarray
        Number density per dex in each bin.
    error : ndarray
        Poisson uncertainty on the number density.
    n_counts : ndarray, optional
        Raw counts in each valid bin (only if *return_n_counts* is ``True``).

    .. note::
       Unit handling is currently the caller's responsibility — ensure
       *masses*, *box_volume*, and the mass limits share consistent units.
    """
    masses = np.ascontiguousarray(masses)

    log_low, log_high = np.log10(lowest_mass), np.log10(highest_mass)
    bins = np.logspace(log_low, log_high, n_bins + 1)

    bin_width_dex = (log_high - log_low) / n_bins
    normalization_factor = 1.0 / (bin_width_dex * box_volume)

    counts, _ = np.histogram(masses, bins)
    valid = counts >= minimum_in_bin
    counts_valid = counts[valid]

    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    result = (
        bin_centers[valid],
        counts_valid * normalization_factor,
        np.sqrt(counts_valid) * normalization_factor,
    )

    if return_n_counts:
        return result + (counts_valid,)

    return result
