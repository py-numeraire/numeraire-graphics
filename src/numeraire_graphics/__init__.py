"""numeraire-graphics — grammar-of-graphics figures over numeraire's results and Output objects.

plotnine is primary: every plot function returns a ``ggplot`` grammar object (it never draws or
saves), so the caller composes and then saves deliberately via :func:`save_paper`. Two input
families:

- **Family A — result-schema plotters** (:mod:`numeraire_graphics.plots`): consume the tidy
  result table every evaluator emits (``plot_cumulative``, ``plot_rolling``, ``plot_metric_by``,
  ``plot_complexity_curve``, ``plot_ic_decay``).
- **Family B — Output/frame plotters** (:mod:`numeraire_graphics.outputs`): consume a numeraire
  Output object or a caller-supplied frame for the richer inputs the tidy schema does not carry
  (``plot_weights_heatmap``, ``plot_factor_loadings``, ``plot_frontier``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from numeraire_graphics.outputs import (
    mean_variance_frontier,
    plot_factor_loadings,
    plot_frontier,
    plot_weights_heatmap,
)
from numeraire_graphics.plots import (
    plot_complexity_curve,
    plot_cumulative,
    plot_ic_decay,
    plot_metric_by,
    plot_rolling,
)
from numeraire_graphics.theme import (
    save_paper,
    scale_color_numeraire,
    scale_fill_numeraire,
    theme_numeraire,
)

__all__ = [
    "__version__",
    "mean_variance_frontier",
    "plot_complexity_curve",
    "plot_cumulative",
    "plot_factor_loadings",
    "plot_frontier",
    "plot_ic_decay",
    "plot_metric_by",
    "plot_rolling",
    "plot_weights_heatmap",
    "save_paper",
    "scale_color_numeraire",
    "scale_fill_numeraire",
    "theme_numeraire",
]

try:
    __version__ = version("numeraire-graphics")
except PackageNotFoundError:  # pragma: no cover - package not installed (e.g. source tree)
    __version__ = "0.0.0"
