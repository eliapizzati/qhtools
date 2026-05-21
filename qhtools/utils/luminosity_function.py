"""Analytic quasar luminosity function (QLF) helpers.

Migrated from ``swift_qso_model.core_functions.quasar_luminosity_function``.
"""

import numpy as np


def create_qlf_Lbol_from_fit(qlf_params, lowest_logL=9, highest_logL=18, n_points=51):
    """Double-power-law bolometric QLF evaluated on a log-L grid.

    phi(L) = phi_star / [ (L/L_star)^gamma_1 + (L/L_star)^gamma_2 ]

    Parameters
    ----------
    qlf_params : sequence of 4 floats
        ``[log10(phi_star), log10(L_star), gamma_1, gamma_2]``.
    lowest_logL, highest_logL : float
        log10 bolometric-luminosity range (erg/s).
    n_points : int
        Number of intervals (the grid has ``n_points + 1`` points).

    Returns
    -------
    log_L_bol : ndarray, shape (n_points + 1,)
        log10 bolometric luminosity grid.
    log_phi : ndarray, shape (n_points + 1,)
        log10 number density per dex.
    """
    log_L_bol = np.linspace(lowest_logL, highest_logL, n_points + 1)

    log10phi_star = qlf_params[0]
    log_L_star = qlf_params[1]
    gamma_1 = qlf_params[2]
    gamma_2 = qlf_params[3]

    phi = np.power(10, log10phi_star) / (
        (np.power(10, log_L_bol) / np.power(10, log_L_star)) ** gamma_1
        + (np.power(10, log_L_bol) / np.power(10, log_L_star)) ** gamma_2
    )

    return log_L_bol, np.log10(phi)
