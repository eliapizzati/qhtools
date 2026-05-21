"""Chi-squared between a binned model curve and a data object.

Migrated verbatim (with a tiny tidy) from
``swift_qso_model.core_functions.inference_global.get_chi2`` so that qhtools is a
self-contained home for the inference helpers.
"""

import numpy as np


def get_chi2(data, x_axis, y_axis, print_info=False):
    """Chi-squared of a model curve ``(x_axis, y_axis)`` against ``data``.

    The model is linearly interpolated onto the data abscissae ``data.x`` and
    compared either with a full covariance matrix (if ``data.covariance`` is not
    ``None``) or with diagonal errors ``data.err``.

    Parameters
    ----------
    data : object
        Duck-typed data container. Must expose:
          - ``x``           : 1D array of abscissae where the data is measured
          - ``data``        : 1D array of measured values
          - ``err``         : 1D array of 1-sigma errors (diagonal case)
          - ``covariance``  : 2D array or ``None``. If not ``None`` the full
                              covariance is used and ``err`` is ignored.
          - ``label``       : only read when ``print_info=True``
    x_axis, y_axis : 1D arrays
        Model curve. ``y_axis`` is interpolated onto ``data.x`` via
        ``np.interp`` (so ``x_axis`` must be sorted ascending).
    print_info : bool
        If True, print the interpolated model, data, residuals and chi2.

    Returns
    -------
    float
        chi2 = (m - d)^T C^{-1} (m - d) with full covariance, else
        sum(((m - d) / err)^2).
    """
    function = np.interp(data.x, x_axis, y_axis)
    diff = function - data.data

    if data.covariance is not None:
        # Full covariance: chi2 = diff^T C^{-1} diff (solve avoids forming C^{-1})
        chi2 = diff @ np.linalg.solve(data.covariance, diff)
        residuals = None
    else:
        residuals = diff / data.err
        chi2 = np.sum(residuals ** 2)

    if print_info:
        print("getting chi2 for ", data.label, " chi2 = ", chi2)
        print("interp func", function)
        print("data+err", data.data, data.err)
        print("residuals+normalized", diff, residuals)

    return chi2
