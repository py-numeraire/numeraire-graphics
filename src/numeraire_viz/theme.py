"""Presentation helpers: a house theme, a colourblind colour scale, and a paper-exact save.

Kept separate from the plot builders because they are composed onto *any* returned grammar
object (``plot + theme_numeraire()``), and because saving is a deliberate, explicit act — the
plot functions never write files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plotnine import (
    element_line,
    element_rect,
    element_text,
    scale_color_manual,
    theme,
    theme_matplotlib,
)

from numeraire_viz._common import OKABE_ITO

# cm-per-inch, for the paper-exact save path.
_CM_PER_INCH = 2.54

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


def theme_numeraire(base_size: float = 8.0, base_family: str = "serif") -> theme:
    """The house theme: a clean, publication-oriented look on a matplotlib base.

    ``base_size`` (points) and ``base_family`` set the typographic defaults; 8pt serif suits a
    single-column journal figure. Returns a plotnine ``theme`` to add onto any plot.
    """
    return theme_matplotlib() + theme(
        text=element_text(family=base_family, size=base_size),
        axis_title=element_text(size=base_size),
        legend_title=element_text(size=base_size),
        plot_title=element_text(size=base_size + 1),
        panel_grid_major=element_line(color="#E6E6E6", size=0.3),
        panel_grid_minor=element_line(color="#F2F2F2", size=0.2),
        panel_background=element_rect(fill="white"),
        legend_key=element_rect(fill="white"),
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
        greys = ["#000000", "#555555", "#888888", "#AAAAAA", "#333333", "#777777", "#999999"]
        return scale_color_manual(values=greys, name="")
    return scale_color_manual(values=list(OKABE_ITO), name="")


def save_paper(
    plot: Any,
    path: str | Path,
    *,
    width_cm: float,
    height_cm: float,
    font_profile: str = "latex",
    dpi: int = 300,
) -> Path:
    """Save ``plot`` at an exact centimetre size for a paper, under a print font profile.

    ``width_cm`` / ``height_cm`` size the figure exactly (journals specify column widths in cm);
    ``font_profile`` (``"latex"`` | ``"sans"`` | ``"none"``) sets matplotlib rcParams for the
    duration of the save only, then restores them. Returns the written path. This is the *only*
    save surface — the plot builders never write.
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
            width=width_cm,
            height=height_cm,
            units="cm",
            dpi=dpi,
            verbose=False,
        )
    finally:
        rc.update(saved)
    return out
