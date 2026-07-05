"""Tests for the presentation helpers: theme, colour scale, and the paper-exact save."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
from plotnine import aes, geom_line, ggplot, theme
from plotnine.scales.scale_manual import scale_color_manual

from numeraire_viz import save_paper, scale_color_numeraire, theme_numeraire
from numeraire_viz._common import OKABE_ITO


def test_theme_numeraire_returns_theme():
    assert isinstance(theme_numeraire(), theme)
    assert isinstance(theme_numeraire(base_size=10, base_family="sans-serif"), theme)


def test_scale_color_numeraire_okabe_ito():
    scale = scale_color_numeraire()
    assert isinstance(scale, scale_color_manual)
    assert scale.palette(1)[0] == OKABE_ITO[0]


def test_scale_color_numeraire_greyscale():
    scale = scale_color_numeraire(greyscale=True)
    assert isinstance(scale, scale_color_manual)
    assert scale.palette(1)[0] == "#000000"


def test_scale_color_numeraire_rejects_unknown_palette():
    with pytest.raises(ValueError, match="palette"):
        scale_color_numeraire(palette="viridis")


def _tiny_plot() -> ggplot:
    import pandas as pd

    d = pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.2, 0.15], "m": ["a", "a", "a"]})
    return ggplot(d, aes("x", "y", color="m")) + geom_line()


def test_save_paper_writes_file_at_requested_cm_size(tmp_path):
    out = tmp_path / "fig.png"
    dpi = 100
    returned = save_paper(_tiny_plot(), out, width_cm=8.0, height_cm=6.0, dpi=dpi)
    assert returned == out
    assert out.exists()
    # cm -> inch -> px: 8cm@100dpi = 8/2.54*100 px wide, 6cm tall.
    from PIL import Image

    with Image.open(out) as img:
        width_px, height_px = img.size
    # cm -> px can floor or round in the matplotlib backend; allow a 1px tolerance.
    assert abs(width_px - round(8.0 / 2.54 * dpi)) <= 1
    assert abs(height_px - round(6.0 / 2.54 * dpi)) <= 1


def test_save_paper_rejects_unknown_font_profile(tmp_path):
    with pytest.raises(ValueError, match="font_profile"):
        save_paper(_tiny_plot(), tmp_path / "f.png", width_cm=8, height_cm=6, font_profile="comic")


def test_save_paper_restores_rcparams(tmp_path):
    import matplotlib as mpl

    before = mpl.rcParams["font.family"]
    save_paper(_tiny_plot(), tmp_path / "f.pdf", width_cm=8, height_cm=6, font_profile="latex")
    assert mpl.rcParams["font.family"] == before
