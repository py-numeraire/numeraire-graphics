"""Figure-quality polish tests: smart date axis, heatmap tiling/fill, legend, robustness, theme.

These are the cases that would have caught the rendered defects — an overlapping date axis, the
heatmap's white stripes and missing diverging fill, the redundant bar legend, a silently-ignored
benchmark typo, and a palette running out. Assertions are on the built ``ggplot`` object (its
scales, layers, data) wherever possible, backed by a headless Agg smoke render. All data synthetic.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import warnings

import numpy as np
import pandas as pd
import pytest
from numeraire.core.engine import WeightsOutput
from numeraire.core.evaluators import SharpeEvaluator, StrategyReturnEvaluator
from plotnine import ggplot
from plotnine.scales.scale_xy import scale_x_datetime

from numeraire_graphics import (
    plot_cumulative,
    plot_metric_by,
    plot_rolling,
    plot_weights_heatmap,
    save_paper,
    theme_numeraire,
)
from numeraire_graphics._common import date_breaks_and_labels, thinned_break_labels

# --- fixtures -------------------------------------------------------------------------------------


def _weights(method: str, seed: int, n_dates: int, n_assets: int = 3) -> WeightsOutput:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=n_dates, freq="ME")
    assets = [f"a{i}" for i in range(n_assets)]
    w = pd.DataFrame(np.full((n_dates, n_assets), 1.0 / n_assets), index=dates, columns=assets)
    r = pd.DataFrame(rng.normal(0.008, 0.04, (n_dates, n_assets)), index=dates, columns=assets)
    return WeightsOutput(
        weights=w,
        realized=r,
        method=method,
        config_hash="c",
        data_vintage="s",
        run_id=f"{method}-r",
    )


@pytest.fixture
def long_results() -> pd.DataFrame:
    """A 20-year (240-month) two-method strategy-return table — the span that crowded the axis."""
    ev = StrategyReturnEvaluator()
    return pd.concat(
        [ev.evaluate(_weights(m, s, 240)) for m, s in (("model_a", 1), ("model_b", 2))],
        ignore_index=True,
    )


@pytest.fixture
def many_method_results() -> pd.DataFrame:
    """A ten-method summary table (exceeds the 8-colour palette) for the overflow-warning test."""
    sharpe = SharpeEvaluator()
    return pd.concat(
        [sharpe.evaluate(_weights(f"m{i}", i, 24)) for i in range(10)], ignore_index=True
    )


@pytest.fixture
def long_weights() -> WeightsOutput:
    """A 15-year monthly long/short weight stream (irregular signs) for the heatmap tiling test."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2005-01-31", periods=180, freq="ME")
    assets = [f"a{i}" for i in range(6)]
    w = pd.DataFrame(rng.normal(0.0, 0.3, (180, 6)), index=dates, columns=assets)
    r = pd.DataFrame(rng.normal(0.0, 0.04, (180, 6)), index=dates, columns=assets)
    return WeightsOutput(
        weights=w, realized=r, method="ls", config_hash="c", data_vintage="s", run_id="r"
    )


def _x_datetime_scale(plot: ggplot) -> scale_x_datetime:
    for s in plot.scales:
        if isinstance(s, scale_x_datetime):
            return s
    raise AssertionError("no scale_x_datetime on the plot")


def _scale_names(plot: ggplot) -> list[str]:
    return [type(s).__name__ for s in plot.scales]


# --- 1. smart date axis ---------------------------------------------------------------------------


def test_date_breaks_rule_spans():
    # multi-year → year-only labels, cadence scaled so the count stays bounded
    assert date_breaks_and_labels(pd.date_range("2000-01-31", periods=240, freq="ME")) == (
        "3 years",
        "%Y",
    )
    # ~2 years → month-and-year
    breaks, fmt = date_breaks_and_labels(pd.date_range("2000-01-31", periods=24, freq="ME"))
    assert breaks.endswith("months") and fmt == "%b %Y"
    # short span → day-and-month
    breaks, fmt = date_breaks_and_labels(pd.date_range("2000-01-01", periods=20, freq="D"))
    assert breaks.endswith("days") and fmt == "%d %b"
    # empty is tolerated
    assert date_breaks_and_labels(pd.Series([], dtype="datetime64[ns]")) == ("1 year", "%Y")


def test_rolling_applies_bounded_year_axis(long_results):
    plot = plot_rolling(long_results, window=12)
    scale = _x_datetime_scale(plot)
    assert scale.labels.fmt == "%Y"  # compact year-only labels, not full ISO
    # the realised breaks over the 20-year span stay a readable handful (never the crowded run)
    span = (pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-31"))
    n_breaks = len(scale.breaks((pd.Timestamp("2000-01-31"), pd.Timestamp("2019-12-31"))))
    assert 4 <= n_breaks <= 9
    assert span  # span sanity


def test_cumulative_applies_smart_date_axis(long_results, tmp_path):
    plot = plot_cumulative(long_results)
    scale = _x_datetime_scale(plot)
    assert scale.labels.fmt == "%Y"
    out = tmp_path / "cum_long.png"
    plot.save(filename=str(out), width=8, height=6, units="cm", dpi=72, verbose=False)
    assert out.stat().st_size > 0


def test_thinned_break_labels_caps_count():
    labels = [f"L{i}" for i in range(100)]
    shown = thinned_break_labels(labels, target=12)
    assert len(shown) <= 12
    assert set(shown) <= set(labels)
    # a short list is returned whole
    assert thinned_break_labels(["a", "b"]) == ["a", "b"]


# --- 2. weights heatmap: ordinal tiling + default diverging fill ----------------------------------


def test_heatmap_default_diverging_fill(long_weights):
    plot = plot_weights_heatmap(long_weights)
    assert "scale_fill_gradient2" in _scale_names(plot)  # signed weights → diverging by default


def test_heatmap_diverging_false_leaves_fill_to_caller(long_weights):
    plot = plot_weights_heatmap(long_weights, diverging=False)
    assert "scale_fill_gradient2" not in _scale_names(plot)


def test_heatmap_date_axis_is_ordinal_no_gaps(long_weights):
    plot = plot_weights_heatmap(long_weights)
    # discrete x → equal-width abutting tiles (no continuous-datetime white stripes)
    assert "scale_x_discrete" in _scale_names(plot)
    assert isinstance(plot.data["date"].dtype, pd.CategoricalDtype)
    # every rebalance date is a category (equally spaced), so tiles cover the axis with no gaps
    assert len(plot.data["date"].cat.categories) == 180
    # only a thinned subset of dates is actually labelled
    xscale = next(s for s in plot.scales if type(s).__name__ == "scale_x_discrete")
    assert len(xscale.breaks) <= 12


def test_heatmap_empty_raises():
    empty = WeightsOutput(
        weights=pd.DataFrame(),
        realized=pd.DataFrame(),
        method="m",
        config_hash="c",
        data_vintage="s",
        run_id="r",
    )
    with pytest.raises(ValueError, match="empty"):
        plot_weights_heatmap(empty)


# --- 3. plot_metric_by: no redundant fill legend -------------------------------------------------


def test_metric_by_drops_redundant_fill_legend(summary_results):
    plot = plot_metric_by(summary_results, metric="sharpe")
    col = next(layer for layer in plot.layers if type(layer.geom).__name__ == "geom_col")
    assert col.show_legend is False
    # the fill aesthetic is still mapped (coloured bars), just not legended
    assert plot.mapping["fill"] == "method"


# --- 4/5. robustness: benchmark validation + palette overflow -------------------------------------


def test_cumulative_unknown_benchmark_raises(long_results):
    with pytest.raises(ValueError, match="benchmark="):
        plot_cumulative(long_results, benchmark="modle_b")  # typo


def test_cumulative_known_benchmark_ok(long_results):
    plot = plot_cumulative(long_results, benchmark="model_b")
    assert isinstance(plot, ggplot)


def test_palette_overflow_warns(many_method_results):
    with pytest.warns(UserWarning, match="exceed the 8-colour"):
        plot_metric_by(many_method_results, metric="sharpe")


def test_no_warning_within_palette(summary_results):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any palette-overflow warning would fail here
        plot_metric_by(summary_results, metric="sharpe")


# --- 5b. save_paper defaults + explicit format ---------------------------------------------------


def test_save_paper_default_size(tmp_path):
    d = pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.2, 0.15], "m": ["a", "a", "a"]})
    from plotnine import aes, geom_line

    plot = ggplot(d, aes("x", "y", color="m")) + geom_line()
    out = save_paper(plot, tmp_path / "fig.png")  # no width/height passed → defaults
    assert out.exists() and out.stat().st_size > 0


def test_save_paper_explicit_format_overrides_extension(tmp_path):
    d = pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.2, 0.15], "m": ["a", "a", "a"]})
    from plotnine import aes, geom_line

    plot = ggplot(d, aes("x", "y", color="m")) + geom_line()
    out = save_paper(plot, tmp_path / "fig.bin", width_cm=8, height_cm=6, format="png")
    # a PNG signature regardless of the .bin extension
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- 6. theme publication defaults ---------------------------------------------------------------


def test_theme_pins_publication_settings():
    th = theme_numeraire()
    keys = set(th.themeables)
    for k in (
        "strip_background",
        "strip_text",
        "axis_text_x",
        "legend_position",
        "panel_spacing",
        "plot_margin",
        "legend_key",
    ):
        assert k in keys, f"theme_numeraire should pin {k}"


def test_theme_x_axis_rotation_hook():
    th = theme_numeraire(x_axis_rotation=45)
    axis_text_x = th.themeables["axis_text_x"]
    props = axis_text_x.properties
    assert props.get("rotation") == 45
    assert props.get("ha") == "right"


def test_theme_grid_defaults_off_and_opts_in() -> None:
    from plotnine.themes.elements import element_blank, element_line

    from numeraire_graphics import theme_numeraire

    off = theme_numeraire().themeables
    assert isinstance(off["panel_grid_major_y"].theme_element, element_blank)
    assert isinstance(off["panel_grid_major_x"].theme_element, element_blank)

    y = theme_numeraire(grid="y").themeables
    assert isinstance(y["panel_grid_major_y"].theme_element, element_line)
    assert isinstance(y["panel_grid_major_x"].theme_element, element_blank)
