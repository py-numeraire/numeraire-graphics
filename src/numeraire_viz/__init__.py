"""numeraire-viz — grammar-of-graphics figures over numeraire's results and Output objects.

plotnine is primary: every plot function returns a ``ggplot`` grammar object (it never draws or
saves), so the caller composes and then saves deliberately via :func:`save_paper`. Two input
families:

- **Family A — result-schema plotters** (:mod:`numeraire_viz.plots`): consume the tidy result table
  every evaluator emits (``plot_cumulative``, ``plot_rolling``, ``plot_metric_by``,
  ``plot_complexity_curve``, ``plot_ic_decay``).
- **Family B — Output/frame plotters** (:mod:`numeraire_viz.outputs`): consume a numeraire Output
  object or a caller-supplied frame for the richer inputs the tidy schema does not carry
  (``plot_weights_heatmap``, ``plot_factor_loadings``, ``plot_frontier``).
"""

from __future__ import annotations

from numeraire_viz.outputs import (
    mean_variance_frontier,
    plot_factor_loadings,
    plot_frontier,
    plot_weights_heatmap,
)
from numeraire_viz.plots import (
    plot_complexity_curve,
    plot_cumulative,
    plot_ic_decay,
    plot_metric_by,
    plot_rolling,
)
from numeraire_viz.theme import (
    save_paper,
    scale_color_numeraire,
    scale_fill_numeraire,
    theme_numeraire,
)

__all__ = [
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

__version__ = "0.1.0"
