"""
Correctness and performance tests for qhmf_to_corr.

Correctness: verify the refactored functions match a pure-numpy reference
implementation (the original algorithm, unrolled without numba).

Performance: time the JIT-compiled functions and compare old vs new approach.
"""

import time
import numpy as np
import pytest


# ---------------------------------------------------------------------------
#  Fixtures: realistic-ish synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Create synthetic triangle + QHMF data resembling real usage."""
    rng = np.random.default_rng(42)

    N_mass = 20       # mass bins in triangle
    R_sep = 30        # separation bins
    N_m_axis = 200    # fine mass grid for QHMF

    log_mbins_centers = np.linspace(10.0, 14.0, N_mass)
    log_rbins_centers = np.linspace(-1.0, 2.0, R_sep)
    log_m_axis = np.linspace(9.5, 14.5, N_m_axis)

    # Power-law-ish triangle: ξ_{ij}(r) ~ (m_i * m_j)^0.3 * r^{-1.8}
    r_vals = 10.0 ** log_rbins_centers
    m_vals = 10.0 ** log_mbins_centers
    triangle = np.empty((N_mass, N_mass, R_sep))
    for i in range(N_mass):
        for j in range(N_mass):
            triangle[i, j, :] = (m_vals[i] * m_vals[j] / 1e24) ** 0.3 * (r_vals / 5.0) ** -1.8

    # Gaussian QHMF peaked at log M = 12
    qhmf = np.exp(-0.5 * ((log_m_axis - 12.0) / 0.5) ** 2)
    qhmf /= np.trapz(qhmf, x=log_m_axis)

    # Second QHMF for cross-correlation (shifted peak)
    qhmf2 = np.exp(-0.5 * ((log_m_axis - 11.5) / 0.7) ** 2)
    qhmf2 /= np.trapz(qhmf2, x=log_m_axis)

    return dict(
        log_rbins_centers=log_rbins_centers,
        log_mbins_centers=log_mbins_centers,
        triangle=triangle,
        log_m_axis=log_m_axis,
        qhmf=qhmf,
        qhmf2=qhmf2,
    )


# ---------------------------------------------------------------------------
#  Reference implementation (pure numpy, mirrors original algorithm exactly)
# ---------------------------------------------------------------------------

def _reference_auto(log_rbins_centers, log_mbins_centers, triangle,
                    log_m_axis, qhmf):
    """Original algorithm reimplemented in pure numpy for verification."""
    delta = log_mbins_centers[1] - log_mbins_centers[0]
    xi_total = np.zeros_like(log_rbins_centers)
    norm_tot = np.trapz(qhmf, x=log_m_axis)
    if norm_tot == 0.0:
        return log_rbins_centers, xi_total

    for i in range(len(log_mbins_centers)):
        for j in range(len(log_mbins_centers)):
            if log_mbins_centers[i] < log_m_axis[0]:
                norm_i = 0.0
            else:
                i_lo = np.searchsorted(log_m_axis, log_mbins_centers[i] - delta / 2.0)
                i_hi = np.searchsorted(log_m_axis, log_mbins_centers[i] + delta / 2.0)
                norm_i = np.mean(qhmf[i_lo:i_hi]) * delta

            if log_mbins_centers[j] < log_m_axis[0]:
                norm_j = 0.0
            else:
                j_lo = np.searchsorted(log_m_axis, log_mbins_centers[j] - delta / 2.0)
                j_hi = np.searchsorted(log_m_axis, log_mbins_centers[j] + delta / 2.0)
                norm_j = np.mean(qhmf[j_lo:j_hi]) * delta

            xi_total += triangle[i, j, :] * (norm_i / norm_tot) * (norm_j / norm_tot)

    return log_rbins_centers, xi_total


def _reference_cross(log_rbins_centers, log_mbins_centers, triangle,
                     log_m_axis, qhmf1, qhmf2):
    """Original cross-correlation algorithm for verification."""
    delta = log_mbins_centers[1] - log_mbins_centers[0]
    xi_total = np.zeros_like(log_rbins_centers)
    norm1 = np.trapz(qhmf1, x=log_m_axis)
    norm2 = np.trapz(qhmf2, x=log_m_axis)
    if norm1 == 0.0 or norm2 == 0.0:
        return log_rbins_centers, xi_total

    for i in range(len(log_mbins_centers)):
        for j in range(len(log_mbins_centers)):
            if log_mbins_centers[i] < log_m_axis[0]:
                norm_i = 0.0
            else:
                i_lo = np.searchsorted(log_m_axis, log_mbins_centers[i] - delta / 2.0)
                i_hi = np.searchsorted(log_m_axis, log_mbins_centers[i] + delta / 2.0)
                norm_i = np.mean(qhmf1[i_lo:i_hi]) * delta

            if log_mbins_centers[j] < log_m_axis[0]:
                norm_j = 0.0
            else:
                j_lo = np.searchsorted(log_m_axis, log_mbins_centers[j] - delta / 2.0)
                j_hi = np.searchsorted(log_m_axis, log_mbins_centers[j] + delta / 2.0)
                norm_j = np.mean(qhmf2[j_lo:j_hi]) * delta

            xi_total += triangle[i, j, :] * (norm_i / norm1) * (norm_j / norm2)

    return log_rbins_centers, xi_total


# ---------------------------------------------------------------------------
#  Correctness tests
# ---------------------------------------------------------------------------

class TestCorrectness:

    def test_auto_matches_reference(self, synthetic_data):
        from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle

        _, xi_new = get_corr_from_triangle(**{
            k: synthetic_data[k]
            for k in ['log_rbins_centers', 'log_mbins_centers', 'triangle',
                       'log_m_axis', 'qhmf']
        })
        _, xi_ref = _reference_auto(**{
            k: synthetic_data[k]
            for k in ['log_rbins_centers', 'log_mbins_centers', 'triangle',
                       'log_m_axis', 'qhmf']
        })
        np.testing.assert_allclose(xi_new, xi_ref, rtol=1e-12)

    def test_cross_matches_reference(self, synthetic_data):
        from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle_cross

        _, xi_new = get_corr_from_triangle_cross(
            synthetic_data['log_rbins_centers'],
            synthetic_data['log_mbins_centers'],
            synthetic_data['triangle'],
            synthetic_data['log_m_axis'],
            synthetic_data['qhmf'],
            synthetic_data['qhmf2'],
        )
        _, xi_ref = _reference_cross(
            synthetic_data['log_rbins_centers'],
            synthetic_data['log_mbins_centers'],
            synthetic_data['triangle'],
            synthetic_data['log_m_axis'],
            synthetic_data['qhmf'],
            synthetic_data['qhmf2'],
        )
        np.testing.assert_allclose(xi_new, xi_ref, rtol=1e-12)

    def test_backward_compat_alias(self, synthetic_data):
        from qhtools.clustering.qhmf_to_corr import (
            get_corr_from_triangle_cross,
            get_corr_from_triangle_different_qhmfs,
        )
        assert get_corr_from_triangle_different_qhmfs is get_corr_from_triangle_cross

    def test_zero_qhmf_returns_zeros(self, synthetic_data):
        from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle

        zero_qhmf = np.zeros_like(synthetic_data['qhmf'])
        _, xi = get_corr_from_triangle(
            synthetic_data['log_rbins_centers'],
            synthetic_data['log_mbins_centers'],
            synthetic_data['triangle'],
            synthetic_data['log_m_axis'],
            zero_qhmf,
        )
        assert np.all(xi == 0.0)

    def test_delta_qhmf_selects_single_bin(self, synthetic_data):
        """A delta-function QHMF at bin k should give xi ~ triangle[k, k, :]."""
        from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle

        d = synthetic_data
        # Put all mass in a single bin
        k = 10  # pick a bin in the middle
        target_m = d['log_mbins_centers'][k]
        delta_qhmf = np.exp(-0.5 * ((d['log_m_axis'] - target_m) / 0.01) ** 2)

        _, xi = get_corr_from_triangle(
            d['log_rbins_centers'], d['log_mbins_centers'], d['triangle'],
            d['log_m_axis'], delta_qhmf,
        )
        # Should be dominated by triangle[k, k, :] (not exact due to
        # discrete bin width of the delta approximation)
        np.testing.assert_allclose(xi, d['triangle'][k, k, :], rtol=0.1)


# ---------------------------------------------------------------------------
#  Performance tests
# ---------------------------------------------------------------------------

class TestPerformance:

    def test_jit_speedup_over_reference(self, synthetic_data):
        """Verify the numba version is faster than pure-python reference."""
        from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle

        d = synthetic_data
        args = (d['log_rbins_centers'], d['log_mbins_centers'], d['triangle'],
                d['log_m_axis'], d['qhmf'])

        # Warm up JIT
        get_corr_from_triangle(*args)

        n_iter = 200

        t0 = time.perf_counter()
        for _ in range(n_iter):
            get_corr_from_triangle(*args)
        t_new = (time.perf_counter() - t0) / n_iter

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _reference_auto(*args)
        t_ref = (time.perf_counter() - t0) / n_iter

        speedup = t_ref / t_new
        print(f"\n  Numba: {t_new*1e6:.0f} µs | Reference: {t_ref*1e6:.0f} µs | "
              f"Speedup: {speedup:.1f}x")
        # Numba should be meaningfully faster
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"

    def test_weight_precomputation_benefit(self, synthetic_data):
        """Show that precomputing weights (O(N) instead of O(N²)) matters
        as mass bins grow."""
        from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle

        # Scale up: 50 mass bins
        N_mass = 50
        R_sep = 30
        N_m_axis = 500
        log_mbins = np.linspace(10.0, 14.0, N_mass)
        log_rbins = np.linspace(-1.0, 2.0, R_sep)
        log_m_axis = np.linspace(9.5, 14.5, N_m_axis)
        triangle = np.ones((N_mass, N_mass, R_sep))
        qhmf = np.exp(-0.5 * ((log_m_axis - 12.0) / 0.5) ** 2)

        args = (log_rbins, log_mbins, triangle, log_m_axis, qhmf)

        # Warm up
        get_corr_from_triangle(*args)
        _reference_auto(*args)

        n_iter = 100

        t0 = time.perf_counter()
        for _ in range(n_iter):
            get_corr_from_triangle(*args)
        t_new = (time.perf_counter() - t0) / n_iter

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _reference_auto(*args)
        t_ref = (time.perf_counter() - t0) / n_iter

        print(f"\n  50 mass bins — Numba: {t_new*1e6:.0f} µs | "
              f"Reference: {t_ref*1e6:.0f} µs | Speedup: {t_ref/t_new:.1f}x")
