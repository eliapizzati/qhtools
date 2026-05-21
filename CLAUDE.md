# CLAUDE.md — qhtools

## What is qhtools?

A personal Python utility package by Elia Pizzati collecting reusable routines
for quasar/halo cosmology research. Installed as an editable package
(`pip install -e .`) and imported across multiple projects.

## Repository structure

```
qhtools/
├── __init__.py                          # exposes clustering, plotting
├── clustering/
│   ├── __init__.py                      # re-exports the public API of all three clustering modules
│   ├── projected_correlation_functions.py  # wp(rp) and volume-averaged xi from binned xi(r)
│   ├── qhmf_to_corr.py                  # abundance-weighted ξ from triangle + QHMF
│   └── corr_inputs.py                   # load cached HMF-fit + triangle .npz artifacts
├── plotting/
│   └── __init__.py                      # placeholder, no utilities yet
└── utils/
    ├── __init__.py
    ├── cosmology.py                     # shared Colossus cosmology (FLAMINGO default)
    ├── create_binned_functions.py       # luminosity functions & halo mass functions from catalogues
    ├── luminosity_function.py           # analytic double-power-law QLF from fit params
    ├── chi2.py                          # chi2 of a model curve vs a data object (diag or full covariance)
    ├── magnitude_conversion.py          # magnitude / luminosity conversions
    ├── my_utils.py                      # miscellaneous helpers (unit conversion, percentiles, pi_max, M*–Mhalo)
    └── natconst.py                      # CGS physical constants (from RADMC)
```

## Key modules

### `clustering/projected_correlation_functions.py`
Numba-JIT routines that convert a binned 3D correlation function xi(r) into:
- `get_projected_wp(rp, corr, bin_edges, pimax, n_sub)` — projected wp(rp)
- `get_volume_averaged_xi(output_edges, corr, bin_edges, pimax, n_sub)` — volume-averaged xi in cylindrical annuli

Uses power-law sub-binning + piecewise-constant exact integration (closed-form
antiderivatives). Achieves <1e-4 fractional error with default `n_sub=10`.
Numba is justified here — the tight nested loops over bins/sub-bins benefit
heavily from JIT compilation.

Both entry points accept the input ξ radial bins as either edges (length N+1) or
centers (length N) — `_ensure_edges` auto-detects by length and rebuilds edges as
log-space midpoints when centers are passed. This makes the common
`get_corr_from_triangle` → projection pipeline correct without the caller having
to convert (passing centers as edges silently shifts the profile by ~half a bin).

### `clustering/qhmf_to_corr.py`
Computes the effective (abundance-weighted) correlation function from a
precomputed "triangle" array ξ(m_i, m_j, r) and one or two quasar–halo mass
functions (QHMFs). This is the standard HOD abundance-weighted clustering
calculation:

    ξ_eff(r) = Σ_{i,j} ξ_{ij}(r) × w_i × w_j

where w_i = ∫_{bin_i} φ(m) dm / ∫ φ dm is the fractional number density in
mass bin i. Valid under standard HOD assumptions (independent halo occupation,
triangle precomputed per mass-bin pair from simulations or a halo model).

- `get_corr_from_triangle(log_rbins_centers, log_mbins_centers, triangle, log_m_axis, qhmf)` — auto-correlation (single QHMF weights both axes)
- `get_corr_from_triangle_cross(...)` / `get_corr_from_triangle_different_qhmfs(...)` — cross-correlation (independent QHMFs per axis, e.g. quasar–galaxy)

Weights use a rectangle-rule approximation (<φ>_i × Δlog_m) over a fine
log_m_axis grid — accurate when the fine grid (typically 200+ points) is much
denser than the triangle mass bins (~20). Weights are precomputed in O(N) before
the double loop; the original implementation recomputed them at O(N²) cost.
Numba JIT gives ~270x speedup over pure Python at typical sizes.

Normalisation uses `np.trapz`, not `np.trapezoid` — numba supports the former
but not the new name, so the DeprecationWarning under numpy ≥2 is deliberate.

### `clustering/corr_inputs.py`
Portable loader for the correlation-model inputs (HMF fit + 3D "triangle") used
by the clustering likelihood. The artifacts are expensive to build (curve_fit on
simulation HMFs, a 3D polynomial fit on the box triangle, colossus halo-bias
loops) and depend on raw simulation data, so they are produced once offline and
cached as a small `.npz` per `(kind, redshift, resolution, mass-range)` config.
This module is the consumer side — no dependency on the raw data or fitting code.

- `corr_inputs_filename(kind, redshift, len_mbins, len_rbins, log_M_min, log_M_max)` — canonical artifact filename, shared by producer and loader so they agree.
- `get_corr_inputs(data_dir, ...)` — returns the same 5-tuple as the original
  `swift_qso_model.get_input_quantities` `(log_m_axis, rbins, out_bins, mf_fit, triangle_fit)`; raises `FileNotFoundError` listing available configs if no artifact matches.

`kind="auto"` returns `rpbins` (for wp/rp); `kind="cross"` returns `rcross_bins`
(volume-averaged xi). The producer script (`build_corr_inputs.py`) lives next to
the data, e.g. in the swift project's `obs_data/`.

### `utils/create_binned_functions.py`
- `create_luminosity_function(luminosities, box_volume, ...)` — binned quasar luminosity function.
  Default mode is `L_bol` (log-spaced bins); also supports `M_1450` (linear bins).
- `create_mass_function(masses, box_volume, ...)` — binned halo mass function (log-spaced bins).

Both use `np.histogram` for binning, which is already optimal (SIMD-accelerated,
O(n log n) via searchsorted). Custom approaches (bincount with direct index
arithmetic, numba loops) were benchmarked and found to be either slower for
large arrays or have floating-point correctness issues at bin boundaries.

Input arrays are kept in their native dtype (typically float32) via
`np.ascontiguousarray` — forcing float64 was measured to be up to 1.6x slower
for large catalogues due to doubled memory bandwidth.

### `utils/luminosity_function.py`
- `create_qlf_Lbol_from_fit(qlf_params, lowest_logL, highest_logL, n_points)` —
  analytic double-power-law bolometric QLF evaluated on a log-L grid:
  `phi(L) = phi_star / [(L/L_star)^gamma_1 + (L/L_star)^gamma_2]`. Returns
  `(log_L_bol, log_phi)`. Distinct from `create_binned_functions.create_luminosity_function`,
  which bins a *catalogue*; this one evaluates a *fit*.

Migrated from `swift_qso_model.core_functions.quasar_luminosity_function`.

### `utils/chi2.py`
- `get_chi2(data, x_axis, y_axis, print_info=False)` — chi2 of a model curve
  `(x_axis, y_axis)` against a duck-typed `data` object (`.x`, `.data`, `.err`,
  `.covariance`, `.label`). Interpolates the model onto `data.x` with `np.interp`
  (so `x_axis` must be sorted), then uses the full covariance via
  `np.linalg.solve` if `data.covariance is not None`, else diagonal `data.err`.

Migrated from `swift_qso_model.core_functions.inference_global`.

### `utils/my_utils.py`
- `to_ergs(log_L)` / `to_solar(log_L)` — log-luminosity unit conversion (solar <-> erg/s).
- `get_pimax_from_vmax(vmax, redshift, cosmo=None)` — convert a max velocity offset
  (km/s) to a line-of-sight distance pi_max in comoving Mpc (no h). Defaults to the
  package-wide cosmology (imported from `utils/cosmology.py`).
- `get_percentiles(y, x, percentiles)` — percentiles of a distribution y(x) via
  cumulative trapezoidal integration + `np.interp`. O(n) complexity.
- `get_median(y, x)` — thin wrapper calling `get_percentiles` at 0.5.
- `log_mstar_behroozi_18(log_m, z)` — Behroozi+18 (UniverseMachine) halo-mass →
  stellar-mass relation, hard-coded best-fit params; numba-`@jit` so it can be
  called from other compiled routines.

`get_median` was originally O(n^2) (recomputed the integral at every grid point
in a numba loop). Replaced with the O(n) cumulative_trapezoid approach — 1000x+
faster at 10K points. `interp1d` was replaced with `np.interp` for ~2x speedup.

### `utils/cosmology.py`
Wraps Colossus to provide a shared cosmology instance (`cosmo`). Default is
FLAMINGO (3x2pt + all) parameters. Switchable at runtime via `set_cosmology()`.

### `utils/natconst.py`
Physical constants in CGS units, translated from RADMC's IDL routines. Includes
fundamental constants, solar/planetary values, unit conversions, and derived
quantities (Eddington luminosity ratio, etc.).

## Dependencies

- **Runtime**: numpy (>=1.24), numba (>=0.59), scipy, colossus
- **Dev**: pytest, ruff
- Build: setuptools + pyproject.toml

## Performance notes

- Catalogues are typically float32 — do not upcast to float64 unnecessarily.
- `np.histogram` outperforms custom bincount/numba for the histogram workloads
  in this package (memory-bandwidth-bound for large arrays; numpy uses SIMD).
- Numba is used where it genuinely helps: tight nested loops in the clustering
  module (correlation function integration). It does not help for the binned
  function builders or the percentile/median utilities.
- `np.ascontiguousarray` (without dtype coercion) costs ~0.1 us on already-
  contiguous arrays — negligible overhead for the safety it provides.
- `qhmf_to_corr`: mass-bin weights must be precomputed outside the double loop
  (O(N) not O(N²)). The inner loop is a tight multiply-accumulate over the
  separation axis — exactly the pattern where numba excels (~270x over pure
  Python at 20 mass bins × 30 sep bins, ~300x at 50 mass bins).
