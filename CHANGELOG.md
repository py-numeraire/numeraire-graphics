# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
