"""Figure tests: assert on the returned grammar object, plus a headless smoke render.

No display is required — the object is inspected directly (its ``.data``, layers, and mappings),
and each plot is rendered once to a temp file via matplotlib's Agg backend to confirm it draws.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

import pandas as pd
import pytest
from plotnine import ggplot

from numeraire_viz import (
    plot_complexity_curve,
    plot_cumulative,
    plot_ic_decay,
    plot_metric_by,
    plot_rolling,
)


def _geoms(plot: ggplot) -> list[str]:
    return [type(layer.geom).__name__ for layer in plot.layers]


def _layer_mapping(plot: ggplot, geom: str) -> dict:
    for layer in plot.layers:
        if type(layer.geom).__name__ == geom:
            return dict(layer.mapping)
    raise AssertionError(f"no {geom} layer")


def _smoke_render(plot: ggplot, tmp_path, name: str) -> None:
    out = tmp_path / f"{name}.png"
    plot.save(filename=str(out), width=8, height=6, units="cm", dpi=72, verbose=False)
    assert out.exists() and out.stat().st_size > 0


# --- plot_cumulative -------------------------------------------------------------------------


def test_cumulative_returns_ggplot_with_wealth_and_drawdown(strategy_return_results, tmp_path):
    plot = plot_cumulative(strategy_return_results)
    assert isinstance(plot, ggplot)
    assert set(plot.data["panel"].unique()) == {"Cumulative return", "Drawdown"}
    assert "geom_line" in _geoms(plot)
    assert _layer_mapping(plot, "geom_line")["color"] == "method"
    _smoke_render(plot, tmp_path, "cumulative")


def test_cumulative_with_series_benchmark_and_recessions(strategy_return_results, tmp_path):
    bench = pd.Series(
        [0.01] * 60,
        index=pd.date_range("2000-01-31", periods=60, freq="ME"),
        name="equal_weight",
    )
    recessions = [("2001-03-31", "2001-11-30"), ("2007-12-31", "2009-06-30")]
    plot = plot_cumulative(strategy_return_results, benchmark=bench, recessions=recessions)
    assert "equal_weight" in set(plot.data["method"])
    assert "geom_rect" in _geoms(plot)
    _smoke_render(plot, tmp_path, "cumulative_bench")


def test_cumulative_named_benchmark_marks_role(strategy_return_results):
    plot = plot_cumulative(strategy_return_results, benchmark="model_b")
    roles = plot.data.loc[plot.data["method"] == "model_b", "role"].unique()
    assert list(roles) == ["benchmark"]
    assert _layer_mapping(plot, "geom_line")["linetype"] == "role"


# --- plot_rolling ----------------------------------------------------------------------------


def test_rolling_sharpe_returns_line(strategy_return_results, tmp_path):
    plot = plot_rolling(strategy_return_results, window=12)
    assert isinstance(plot, ggplot)
    assert _geoms(plot).count("geom_line") == 1
    assert plot.mapping["color"] == "method"
    # rolling(12) drops the first 11 dates per method (2 methods x (60-11) = 98 rows).
    assert len(plot.data) == 2 * (60 - 11)
    _smoke_render(plot, tmp_path, "rolling")


def test_rolling_rejects_short_window(strategy_return_results):
    with pytest.raises(ValueError, match="window"):
        plot_rolling(strategy_return_results, window=1)


def test_rolling_rejects_unknown_metric(strategy_return_results):
    with pytest.raises(ValueError, match="metric"):
        plot_rolling(strategy_return_results, window=12, metric="omega")


# --- plot_metric_by --------------------------------------------------------------------------


def test_metric_by_method_has_error_bars_from_repeated_rows(summary_results, tmp_path):
    plot = plot_metric_by(summary_results, metric="sharpe")
    assert isinstance(plot, ggplot)
    assert "geom_col" in _geoms(plot)
    assert "geom_errorbar" in _geoms(plot)  # two rows per method -> derived CI
    assert set(plot.data["method"]) == {"model_a", "model_b"}
    _smoke_render(plot, tmp_path, "metric_by")


def test_metric_by_universe_grouping(summary_results):
    plot = plot_metric_by(summary_results, metric="sharpe", x="universe")
    assert plot.mapping["x"] == "universe"
    assert set(plot.data["universe"]) == {"n=25", "n=30"}


def test_metric_by_plain_bars_without_ci():
    rows = [
        {
            "run_id": "r",
            "method": m,
            "date": pd.Timestamp("2010-12-31"),
            "metric": "sharpe",
            "value": v,
            "universe": "n=25",
            "capability": "to_weights",
            "protocol": "walk_forward",
            "config_hash": "c",
            "data_vintage": "synthetic",
        }
        for m, v in (("a", 0.5), ("b", 0.7))
    ]
    plot = plot_metric_by(pd.DataFrame(rows), metric="sharpe")
    assert "geom_errorbar" not in _geoms(plot)  # one row per group -> no CI


def test_metric_by_se_column_builds_ci(summary_results):
    df = summary_results.copy()
    df["se"] = 0.05
    plot = plot_metric_by(df, metric="sharpe")
    assert "geom_errorbar" in _geoms(plot)


def test_metric_by_rejects_missing_group(summary_results):
    with pytest.raises(ValueError, match="grouping column"):
        plot_metric_by(summary_results, metric="sharpe", x="nonexistent")


# --- plot_complexity_curve -------------------------------------------------------------------


def test_complexity_curve_line_and_points(complexity_results, tmp_path):
    plot = plot_complexity_curve(complexity_results, x="complexity", metric="sharpe")
    assert isinstance(plot, ggplot)
    geoms = _geoms(plot)
    assert "geom_line" in geoms and "geom_point" in geoms
    assert plot.mapping["x"] == "complexity"
    # sorted along the complexity axis
    assert list(plot.data["complexity"]) == sorted(plot.data["complexity"])
    _smoke_render(plot, tmp_path, "complexity")


def test_complexity_curve_with_ribbon(complexity_results, tmp_path):
    plot = plot_complexity_curve(
        complexity_results, x="complexity", metric="sharpe", ribbon=("lo", "hi")
    )
    assert "geom_ribbon" in _geoms(plot)
    _smoke_render(plot, tmp_path, "complexity_ribbon")


def test_complexity_curve_rejects_missing_axis(complexity_results):
    with pytest.raises(ValueError, match="complexity axis"):
        plot_complexity_curve(complexity_results, x="not_joined", metric="sharpe")


def test_complexity_curve_rejects_missing_ribbon_column(complexity_results):
    with pytest.raises(ValueError, match="ribbon column"):
        plot_complexity_curve(
            complexity_results, x="complexity", metric="sharpe", ribbon=("lo", "missing")
        )


# --- plot_ic_decay (family A) ----------------------------------------------------------------


def test_ic_decay_line_and_points_by_method(ic_decay_results, tmp_path):
    plot = plot_ic_decay(ic_decay_results)
    assert isinstance(plot, ggplot)
    geoms = _geoms(plot)
    assert "geom_line" in geoms and "geom_point" in geoms
    assert "geom_hline" in geoms  # zero reference
    assert plot.mapping["x"] == "horizon"
    assert plot.mapping["color"] == "method"
    # sorted along the horizon axis
    assert list(plot.data["horizon"]) == sorted(plot.data["horizon"])
    assert set(plot.data["method"]) == {"voc", "hist_mean"}
    _smoke_render(plot, tmp_path, "ic_decay")


def test_ic_decay_smooth_adds_layer(ic_decay_results, tmp_path):
    plot = plot_ic_decay(ic_decay_results, smooth=True)
    assert "geom_smooth" in _geoms(plot)
    _smoke_render(plot, tmp_path, "ic_decay_smooth")


def test_ic_decay_custom_metric_ic_ir(ic_decay_results):
    df = ic_decay_results.copy()
    df["metric"] = "ic_ir"
    plot = plot_ic_decay(df, metric="ic_ir")
    assert (plot.data["metric"] == "ic_ir").all()


def test_ic_decay_rejects_missing_horizon(ic_decay_results):
    df = ic_decay_results.drop(columns=["horizon"])
    with pytest.raises(ValueError, match="horizon axis"):
        plot_ic_decay(df)
