"""Load precomputed correlation-model inputs (HMF fit + halo-model triangle).

The HMF fit and the 3D correlation "triangle" used by the clustering likelihood
are expensive to build (curve_fit on simulation HMF data, a 3D polynomial fit on
a triangle measured from the box, plus colossus halo-bias loops for the cross
case) and depend on raw simulation data that lives only on the machine where it
was produced. Crucially the *output* is small and deterministic for a fixed
(kind, redshift, resolution, mass-range) config, so it is produced once offline
and cached as a small ``.npz`` per config.

This module is the portable consumer side: given a directory of cached ``.npz``
artifacts it returns the exact same 5-tuple as the original producer
(``swift_qso_model``'s ``get_input_quantities``), with no dependency on the raw
simulation data or the fitting code. The artifacts are produced by a separate
script (kept next to the data, e.g. the swift project's
``obs_data/build_corr_inputs.py``).

API
---
get_corr_inputs(data_dir, kind, redshift, len_mbins, len_rbins, log_M_min, log_M_max)
    Returns (log_m_axis, rbins, out_bins, mf_fit, triangle_fit), where out_bins is
    the projected-separation bins: ``rpbins`` for kind="auto" (wp/rp) or
    ``rcross_bins`` for kind="cross" (volume-averaged xi).

corr_inputs_filename(kind, redshift, len_mbins, len_rbins, log_M_min, log_M_max)
    The canonical artifact filename — shared by producer and loader so they agree.
"""

import os

import numpy as np


def corr_inputs_filename(kind, redshift, len_mbins, len_rbins, log_M_min, log_M_max):
    """Canonical artifact filename for one (kind, z, resolution, mass-range) config."""
    return (
        f"corr_inputs_{kind}_z{redshift:.1f}_m{len_mbins}_r{len_rbins}"
        f"_logM{log_M_min:.1f}_{log_M_max:.1f}.npz"
    )


def get_corr_inputs(data_dir, kind, redshift, len_mbins, len_rbins, log_M_min, log_M_max):
    """Load cached correlation-model inputs for one config.

    Parameters
    ----------
    data_dir : str
        Directory holding the cached ``corr_inputs_*.npz`` artifacts.
    kind : {"auto", "cross"}
        "auto" for the projected auto-correlation (wp/rp); "cross" for the
        volume-averaged quasar-galaxy cross-correlation.
    redshift : float
    len_mbins, len_rbins : int
        Triangle grid resolution the artifact was produced at.
    log_M_min, log_M_max : float
        Halo-mass range the HMF/triangle were built for.

    Returns
    -------
    log_m_axis, rbins, out_bins, mf_fit, triangle_fit : ndarrays
        Same 5-tuple as the original ``get_input_quantities``. ``out_bins`` is
        ``rpbins`` (kind="auto") or ``rcross_bins`` (kind="cross").

    Raises
    ------
    FileNotFoundError
        If no artifact matches the requested config — with the list of available
        configs and a pointer to regenerate them.
    """
    fname = corr_inputs_filename(kind, redshift, len_mbins, len_rbins, log_M_min, log_M_max)
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        available = (
            sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
            if os.path.isdir(data_dir)
            else []
        )
        raise FileNotFoundError(
            f"No cached correlation inputs for kind={kind!r} z={redshift} "
            f"len_mbins={len_mbins} len_rbins={len_rbins} "
            f"logM=[{log_M_min}, {log_M_max}].\n"
            f"  Looked for: {path}\n"
            f"  Available : {available or '(none)'}\n"
            f"  Regenerate with the producer script (build_corr_inputs.py) on a "
            f"machine that has the simulation data."
        )
    with np.load(path) as d:
        return (
            d["log_m_axis"],
            d["rbins"],
            d["out_bins"],
            d["mf_fit"],
            d["triangle_fit"],
        )
