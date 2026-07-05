"""Presentation helpers: a house theme, a colourblind colour scale, and a paper-exact save.

Kept separate from the plot builders because they are composed onto *any* returned grammar
object (``plot + theme_numeraire()``), and because saving is a deliberate, explicit act — the
plot functions never write files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plotnine import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    scale_color_manual,
    scale_fill_gradient2,
    scale_fill_manual,
    theme,
    theme_matplotlib,
)

from numeraire_viz._common import OKABE_ITO

# Colourblind-safe diverging endpoints (Okabe-Ito blue / near-white / vermillion), for a signed
# fill centred at zero: blue = negative (short), vermillion = positive (long).
_DIVERGING_LOW = "#0072B2"
_DIVERGING_MID = "#F7F7F7"
_DIVERGING_HIGH = "#D55E00"

# cm-per-inch, for the paper-exact save path.
_CM_PER_INCH = 2.54

# Ordered greys for a monochrome print (shared by the colour and fill greyscale scales).
_GREYS = ("#000000", "#555555", "#888888", "#AAAAAA", "#333333", "#777777", "#999999")

# How many times to cycle a discrete palette so a comparison with more than eight methods still
# renders every group (colours repeat past eight — the plot builders warn when that happens).
_PALETTE_CYCLES = 3


def _cycled(values: tuple[str, ...]) -> list[str]:
    """Repeat a discrete palette so an oversized comparison never runs out of fill/colour values.

    The first ``len(values)`` entries are unchanged (so the leading draws stay the exact Okabe-Ito
    order); only beyond that do colours repeat. The builders emit a warning when this cycling bites.
    """
    return list(values) * _PALETTE_CYCLES


# rcParam profiles for print. ``latex`` selects Computer-Modern-like serif math without requiring
# a TeX installation (a true PGF/usetex export is available to callers who set it themselves and
# have LaTeX installed); ``sans`` is a neutral screen profile; ``none`` leaves rcParams untouched.
_FONT_PROFILES: dict[str, dict[str, Any]] = {
    "latex": {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    },
    "sans": {
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    },
    "none": {},
}


def theme_numeraire(
    base_size: float = 8.0, base_family: str = "serif", x_axis_rotation: float = 0.0
) -> theme:
    """The house theme: a clean, publication-oriented look on a matplotlib base.

    ``base_size`` (points) and ``base_family`` set the typographic defaults; 8pt serif suits a
    single-column journal figure. ``x_axis_rotation`` (degrees) is the rotation hook for the x-axis
    tick labels — 0 by default (the smart date axis keeps them short), raise it (e.g. 30–45) as a
    fallback when a categorical or dense axis still crowds. Returns a plotnine ``theme`` to add onto
    any plot.

    Beyond the typography this pins the publication defaults plotnine otherwise leaves to its
    matplotlib base: a flat light facet strip (no heavy grey block), a bottom legend with no key
    background, sized axis text, breathing room between facet panels, and a small plot margin — so a
    figure looks intentional straight out of the builder without hand-tuning.

    Only the *horizontal* (y) grid is drawn — the value guides a line/bar reader wants; the
    vertical (x) grid is dropped. That keeps the look minimal and, crucially, means the weight
    heatmap's tiled date axis is not crossed by vertical rules that would read as white stripes.
    """
    x_text_ha = "right" if x_axis_rotation else "center"
    return theme_matplotlib() + theme(
        text=element_text(family=base_family, size=base_size),
        axis_title=element_text(size=base_size),
        axis_text=element_text(size=base_size - 1),
        axis_text_x=element_text(size=base_size - 1, ha=x_text_ha, rotation=x_axis_rotation),
        legend_title=element_text(size=base_size),
        legend_text=element_text(size=base_size - 1),
        plot_title=element_text(size=base_size + 1),
        panel_grid_major_y=element_line(color="#E6E6E6", size=0.3),
        panel_grid_minor_y=element_line(color="#F2F2F2", size=0.2),
        panel_grid_major_x=element_blank(),
        panel_grid_minor_x=element_blank(),
        panel_background=element_rect(fill="white"),
        panel_spacing=0.03,
        # A flat, light facet strip instead of plotnine's heavy grey block.
        strip_background=element_rect(fill="#F0F0F0", color="none"),
        strip_text=element_text(size=base_size, color="#1A1A1A"),
        legend_position="bottom",
        legend_key=element_rect(fill="white", color="none"),
        legend_background=element_rect(fill="white", color="none"),
        plot_margin=0.02,
        figure_size=(8.0 / _CM_PER_INCH, 6.0 / _CM_PER_INCH),  # ~8x6 cm; overridden on save_paper
    )


def scale_color_numeraire(
    palette: str = "okabe_ito", greyscale: bool = False
) -> scale_color_manual:
    """A colourblind-safe discrete colour scale (Okabe-Ito by default).

    ``greyscale=True`` collapses to an ordered set of greys for a monochrome print (pair it with
    varied linetypes for separability). ``palette`` currently accepts only ``"okabe_ito"``.
    """
    if palette != "okabe_ito":
        raise ValueError(f"unknown palette {palette!r}; only 'okabe_ito' is available")
    if greyscale:
        return scale_color_manual(values=_cycled(_GREYS), name="")
    return scale_color_manual(values=_cycled(OKABE_ITO), name="")


def scale_fill_numeraire(
    palette: str = "okabe_ito", greyscale: bool = False, diverging: bool = False
) -> Any:
    """The ``fill`` counterpart of :func:`scale_color_numeraire`, discrete or diverging.

    With ``diverging=False`` (the default) this is the discrete Okabe-Ito fill scale for a
    categorical ``fill`` aesthetic — the bars of :func:`~numeraire_viz.plot_metric_by` and any other
    grouped fill. ``greyscale=True`` collapses it to ordered greys for a monochrome print, matching
    :func:`scale_color_numeraire`.

    With ``diverging=True`` it is a *continuous* two-sided fill centred at zero — the scale the
    weight heatmap wants, so a long (positive) and a short (negative) weight read as opposite hues
    with an unsaturated midpoint at zero. The endpoints are colourblind-safe (blue for negative,
    vermillion for positive); ``greyscale=True`` gives a light-to-dark grey ramp through white.
    ``palette`` currently accepts only ``"okabe_ito"``.
    """
    if palette != "okabe_ito":
        raise ValueError(f"unknown palette {palette!r}; only 'okabe_ito' is available")
    if diverging:
        if greyscale:
            return scale_fill_gradient2(
                low="#333333", mid="#FFFFFF", high="#000000", midpoint=0.0, name=""
            )
        return scale_fill_gradient2(
            low=_DIVERGING_LOW, mid=_DIVERGING_MID, high=_DIVERGING_HIGH, midpoint=0.0, name=""
        )
    if greyscale:
        return scale_fill_manual(values=_cycled(_GREYS), name="")
    return scale_fill_manual(values=_cycled(OKABE_ITO), name="")


def save_paper(
    plot: Any,
    path: str | Path,
    *,
    width_cm: float = 8.4,
    height_cm: float = 6.0,
    font_profile: str = "latex",
    dpi: int = 300,
    format: str | None = None,
) -> Path:
    """Save ``plot`` at an exact centimetre size for a paper, under a print font profile.

    ``width_cm`` / ``height_cm`` size the figure exactly (journals specify column widths in cm); the
    defaults (8.4 x 6 cm) are a single journal column, so a bare ``save_paper(plot, path)`` already
    yields a sensibly-proportioned figure. ``font_profile`` (``"latex"`` | ``"sans"`` | ``"none"``)
    sets matplotlib rcParams for the duration of the save only, then restores them. ``format``
    forces the output format (``"pdf"``, ``"png"``, ``"svg"``, …) regardless of the path extension;
    left ``None`` the extension decides. Returns the written path. This is the *only* save surface —
    the plot builders never write.
    """
    import matplotlib as mpl

    if font_profile not in _FONT_PROFILES:
        raise ValueError(
            f"unknown font_profile {font_profile!r}; choose from {sorted(_FONT_PROFILES)}"
        )
    out = Path(path)
    # matplotlib types rcParams keys as a closed Literal; we pass a dynamic set of profile keys
    # deliberately, so view it as a plain mapping for this scoped save/restore.
    rc: Any = mpl.rcParams
    profile = _FONT_PROFILES[font_profile]
    saved = {k: rc[k] for k in profile}
    try:
        rc.update(profile)
        plot.save(
            filename=str(out),
            format=format,
            width=width_cm,
            height=height_cm,
            units="cm",
            dpi=dpi,
            verbose=False,
        )
    finally:
        rc.update(saved)
    return out
