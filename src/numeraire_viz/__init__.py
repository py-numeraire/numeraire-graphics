"""numeraire-viz — grammar-of-graphics figures over the numeraire tidy result schema.

plotnine is primary: every plot function returns a ``ggplot`` grammar object (it never draws or
saves), so the caller composes and then saves deliberately via :func:`save_paper`. The four plots
consume the standard result schema emitted by numeraire's evaluators.
"""

from __future__ import annotations

from numeraire_viz.plots import (
    plot_complexity_curve,
    plot_cumulative,
    plot_metric_by,
    plot_rolling,
)
from numeraire_viz.theme import save_paper, scale_color_numeraire, theme_numeraire

__all__ = [
    "plot_complexity_curve",
    "plot_cumulative",
    "plot_metric_by",
    "plot_rolling",
    "save_paper",
    "scale_color_numeraire",
    "theme_numeraire",
]

__version__ = "0.1.0"
