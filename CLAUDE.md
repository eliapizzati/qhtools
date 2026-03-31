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
│   ├── __init__.py                      # re-exports get_projected_wp, get_volume_averaged_xi
│   └── projected_correlation_functions.py  # wp(rp) and volume-averaged xi from binned xi(r)
├── plotting/
│   └── __init__.py                      # placeholder, no utilities yet
└── utils/
    ├── __init__.py
    ├── cosmology.py                     # shared Colossus cosmology (FLAMINGO default)
    ├── create_binned_functions.py       # luminosity functions & halo mass functions from catalogues
    ├── my_utils.py                      # miscellaneous helpers (unit conversion, percentiles)
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

### `utils/my_utils.py`
- `to_ergs(log_L)` / `to_solar(log_L)` — log-luminosity unit conversion (solar <-> erg/s).
- `get_percentiles(y, x, percentiles)` — percentiles of a distribution y(x) via
  cumulative trapezoidal integration + `np.interp`. O(n) complexity.
- `get_median(y, x)` — thin wrapper calling `get_percentiles` at 0.5.

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
