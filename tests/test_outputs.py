"""Tests for the Output/frame-consuming figures (family B) and the fill scale / frontier helper.

Each plot is asserted on the returned grammar object (``.data``, layers, mappings) with no display,
plus one headless Agg smoke render. The weights heatmap is exercised against *real* numeraire
``WeightsOutput`` / ``PanelWeightsOutput`` objects (from ``conftest``); loadings and frontier frames
are hand-crafted. All data is synthetic — no external or licensed data.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot
from plotnine.scales.scale_color import scale_fill_gradient2
from plotnine.scales.scale_manual import scale_fill_manual

from numeraire_viz import (
    mean_variance_frontier,
    plot_factor_loadings,
    plot_frontier,
    plot_weights_heatmap,
    scale_fill_numeraire,
)


def _geoms(plot: ggplot) -> list[str]:
    return [type(layer.geom).__name__ for layer in plot.layers]


def _smoke_render(plot: ggplot, tmp_path, name: str) -> None:
    out = tmp_path / f"{name}.png"
    plot.save(filename=str(out), width=8, height=6, units="cm", dpi=72, verbose=False)
    assert out.exists() and out.stat().st_size > 0


# --- plot_weights_heatmap ---------------------------------------------------------------------


def test_weights_heatmap_from_wide_output(weights_output, tmp_path):
    plot = plot_weights_heatmap(weights_output)
    assert isinstance(plot, ggplot)
    assert "geom_tile" in _geoms(plot)
    assert plot.mapping["fill"] == "weight"
    assert set(plot.data.columns) >= {"date", "asset", "weight"}
    # 8 dates x 5 assets = 40 tiles
    assert len(plot.data) == 8 * 5
    _smoke_render(plot, tmp_path, "weights_heatmap")


def test_weights_heatmap_from_panel_output(panel_weights_output, tmp_path):
    plot = plot_weights_heatmap(panel_weights_output)
    assert isinstance(plot, ggplot)
    assert "geom_tile" in _geoms(plot)
    # ragged panel: one row per (date, asset) key
    assert len(plot.data) == len(panel_weights_output.weights)
    _smoke_render(plot, tmp_path, "weights_heatmap_panel")


def test_weights_heatmap_top_keeps_largest_abs(weights_output):
    plot = plot_weights_heatmap(weights_output, top=2)
    assert plot.data["asset"].nunique() == 2
    # the two kept names are those with the largest average absolute weight
    long = weights_output.weights.abs().mean().sort_values(ascending=False)
    assert set(plot.data["asset"].astype(str)) == set(long.index[:2])


def test_weights_heatmap_order_name_is_categorical(weights_output):
    plot = plot_weights_heatmap(weights_output, order="name")
    cats = list(plot.data["asset"].cat.categories)
    assert cats == sorted(cats, reverse=True)


def test_weights_heatmap_rejects_bad_order(weights_output):
    with pytest.raises(ValueError, match="order"):
        plot_weights_heatmap(weights_output, order="sideways")


def test_weights_heatmap_rejects_non_output():
    with pytest.raises(TypeError, match="WeightsOutput"):
        plot_weights_heatmap(pd.DataFrame({"a": [1]}))


def test_weights_heatmap_rejects_nonpositive_top(weights_output):
    with pytest.raises(ValueError, match="top must be"):
        plot_weights_heatmap(weights_output, top=0)


# --- plot_factor_loadings ---------------------------------------------------------------------


@pytest.fixture
def loadings_paths() -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=6, freq="ME")
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "factor": ["f1"] * 6 + ["f2"] * 6,
            "loading": rng.normal(0.0, 1.0, size=12),
        }
    )


@pytest.fixture
def loadings_matrix() -> pd.DataFrame:
    chars = ["size", "value", "mom"]
    rng = np.random.default_rng(4)
    rows = []
    for f in ("f1", "f2"):
        for ch in chars:
            rows.append({"entity": ch, "factor": f, "loading": float(rng.normal())})
    return pd.DataFrame(rows)


def test_factor_loadings_paths(loadings_paths, tmp_path):
    plot = plot_factor_loadings(loadings_paths, x="date")
    assert isinstance(plot, ggplot)
    geoms = _geoms(plot)
    assert "geom_line" in geoms and "geom_point" in geoms and "geom_hline" in geoms
    assert plot.mapping["x"] == "date"
    assert plot.mapping["color"] == "factor"
    _smoke_render(plot, tmp_path, "loadings_paths")


def test_factor_loadings_heatmap_when_no_axis(loadings_matrix, tmp_path):
    plot = plot_factor_loadings(loadings_matrix)
    assert "geom_tile" in _geoms(plot)
    assert plot.mapping["x"] == "entity"
    assert plot.mapping["fill"] == "loading"
    _smoke_render(plot, tmp_path, "loadings_heatmap")


def test_factor_loadings_rejects_malformed_frame():
    with pytest.raises(ValueError, match="missing required column"):
        plot_factor_loadings(pd.DataFrame({"date": [1], "beta": [0.5]}))


def test_factor_loadings_rejects_non_frame():
    with pytest.raises(TypeError, match="tidy DataFrame"):
        plot_factor_loadings([1, 2, 3])  # type: ignore[arg-type]


def test_factor_loadings_rejects_missing_axis(loadings_paths):
    with pytest.raises(ValueError, match="loading axis"):
        plot_factor_loadings(loadings_paths, x="not_a_column")


def test_factor_loadings_heatmap_needs_identifier():
    frame = pd.DataFrame({"factor": ["f1", "f2"], "loading": [0.1, 0.2]})
    with pytest.raises(ValueError, match="entity.*date|heatmap"):
        plot_factor_loadings(frame)


# --- plot_frontier + mean_variance_frontier ---------------------------------------------------


@pytest.fixture
def frontier_frame() -> pd.DataFrame:
    mu = np.array([0.05, 0.08, 0.12])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.06, 0.01], [0.0, 0.01, 0.09]])
    return mean_variance_frontier(mu, cov, n=25)


def test_mean_variance_frontier_shape_and_monotone(frontier_frame):
    assert list(frontier_frame.columns) == ["risk", "return"]
    assert len(frontier_frame) == 25
    assert (frontier_frame["risk"] > 0).all()
    # returns span the asset means, ascending
    assert list(frontier_frame["return"]) == sorted(frontier_frame["return"])


def test_mean_variance_frontier_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="square"):
        mean_variance_frontier(np.array([0.1, 0.2]), np.eye(3))


def test_mean_variance_frontier_rejects_tiny_n():
    with pytest.raises(ValueError, match="n must be"):
        mean_variance_frontier(np.array([0.1, 0.2]), np.eye(2), n=1)


def test_frontier_curve(frontier_frame, tmp_path):
    plot = plot_frontier(frontier_frame)
    assert isinstance(plot, ggplot)
    geoms = _geoms(plot)
    assert "geom_line" in geoms and "geom_point" in geoms
    # sorted along risk for a clean curve
    assert list(plot.data["risk"]) == sorted(plot.data["risk"])
    _smoke_render(plot, tmp_path, "frontier")


def test_frontier_with_labelled_points(frontier_frame, tmp_path):
    points = pd.DataFrame({"risk": [0.22, 0.30], "return": [0.07, 0.11], "label": ["1/N", "GMV"]})
    plot = plot_frontier(frontier_frame, points=points)
    geoms = _geoms(plot)
    assert geoms.count("geom_point") == 2  # frontier points + overlay markers
    assert "geom_text" in geoms
    _smoke_render(plot, tmp_path, "frontier_points")


def test_frontier_points_without_label_no_text(frontier_frame):
    points = pd.DataFrame({"risk": [0.22], "return": [0.07]})
    plot = plot_frontier(frontier_frame, points=points)
    assert "geom_text" not in _geoms(plot)


def test_frontier_rejects_malformed_frame():
    with pytest.raises(ValueError, match="missing required column"):
        plot_frontier(pd.DataFrame({"vol": [0.1], "return": [0.05]}))


def test_frontier_rejects_non_frame():
    with pytest.raises(TypeError, match="must be a DataFrame"):
        plot_frontier([(0.1, 0.05)])  # type: ignore[arg-type]


def test_mean_variance_frontier_rejects_degenerate():
    # identical means make B*C - A^2 == 0 (a degenerate efficient set)
    with pytest.raises(ValueError, match="degenerate"):
        mean_variance_frontier(np.array([0.1, 0.1]), np.eye(2))


# --- scale_fill_numeraire ---------------------------------------------------------------------


def test_scale_fill_numeraire_discrete_okabe_ito():
    from numeraire_viz._common import OKABE_ITO

    scale = scale_fill_numeraire()
    assert isinstance(scale, scale_fill_manual)
    assert scale.palette(1)[0] == OKABE_ITO[0]


def test_scale_fill_numeraire_greyscale_discrete():
    scale = scale_fill_numeraire(greyscale=True)
    assert isinstance(scale, scale_fill_manual)
    assert scale.palette(1)[0] == "#000000"


def test_scale_fill_numeraire_diverging_is_gradient2():
    scale = scale_fill_numeraire(diverging=True)
    assert isinstance(scale, scale_fill_gradient2)


def test_scale_fill_numeraire_diverging_greyscale():
    scale = scale_fill_numeraire(diverging=True, greyscale=True)
    assert isinstance(scale, scale_fill_gradient2)


def test_scale_fill_numeraire_rejects_unknown_palette():
    with pytest.raises(ValueError, match="palette"):
        scale_fill_numeraire(palette="viridis")


def test_weights_heatmap_composes_with_diverging_fill(weights_output, tmp_path):
    # diverging=False leaves the fill scale to the caller; composing one still renders.
    plot = plot_weights_heatmap(weights_output, diverging=False) + scale_fill_numeraire(
        diverging=True
    )
    _smoke_render(plot, tmp_path, "weights_heatmap_diverging")
