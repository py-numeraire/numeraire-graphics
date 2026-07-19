# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Per-date plots now reject non-finite values, duplicate method/date rows, mixed run identities,
  and misaligned comparison calendars instead of silently combining or imputing observations.
- Summary bars require one explicitly aggregated row per group; confidence intervals must come
  from declared bounds or standard errors rather than treating heterogeneous rows as IID samples.
- Factor-loading paths group by factor and entity, heatmaps use a zero-centred diverging scale,
  and mean-variance helpers emit and require the upper efficient frontier branch.

### Fixed

- `plot_cumulative` and `plot_rolling` now raise a clear `ValueError` when the result table yields
  no plottable per-date series (including when `groupby` drops an all-NaN `method` key), instead of
  surfacing a bare `IndexError` from the benchmark branch or a cryptic concat error.
- Made `save_paper` compatible with the Matplotlib 3.11 automatic-backend sentinel exposed through
  plotnine 0.15 theme snapshots.

## [0.1.1] - 2026-07-07

Documentation refresh only — no functional changes.

### Added

- A Sphinx documentation site at <https://graphics.py-numeraire.org/>, with an intersphinx bridge
  into numeraire so `{class}` references resolve onto the parent docs.

### Changed

- Refreshed the README with badges, an ecosystem cross-link, and a citation pointer.

## [0.1.0] - 2026-07-06

### Changed

- **Renamed the package from `numeraire-viz` to `numeraire-graphics`** (distribution name) and
  `numeraire_viz` to `numeraire_graphics` (import name), adopting the project's academic naming
  register. Update imports accordingly: `from numeraire_graphics import ...`.
- Switched to `hatch-vcs` dynamic versioning (tag-driven, matching `numeraire` and
  `numeraire-dataset`); the version is no longer hard-coded. `__version__` is now derived from the
  installed package metadata via `importlib.metadata`.

### Added

- Shipped `py.typed` and the `Typing :: Typed` classifier, marking the package as type-checked.
- Per-version `Programming Language :: Python :: 3.11/3.12/3.13` classifiers and an `Issues`
  project URL.

### Notes

- This is a new, independently versioned package: its **first tagged release will be `0.1.0`**
  (it does not track `numeraire`'s version line). Until a tag is cut, `hatch-vcs` yields a local
  development version.
