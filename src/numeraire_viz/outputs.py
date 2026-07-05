"""The Output/frame-consuming figures (input family B).

The result-schema plotters in :mod:`numeraire_viz.plots` (family A) read the tidy result table
every evaluator emits. These figures need richer inputs the tidy schema deliberately does not
carry — a per-date x asset weight stream, a factor-loadings panel, a risk-return frontier trace —
so they take a numeraire **Output object** or a caller-supplied **frame** directly. The contract is
otherwise identical to family A: each builder returns a plotnine ``ggplot`` and never draws or
saves; the caller composes ``+ theme_numeraire() + scale_fill_numeraire(...)`` and hands the result
to :func:`numeraire_viz.save_paper`.

- :func:`plot_weights_heatmap` — a ``WeightsOutput`` / ``PanelWeightsOutput`` object → a
  date x asset weight matrix as tiles, a diverging fill centred at zero (compose with
  ``scale_fill_numeraire(diverging=True)``).
- :func:`plot_factor_loadings` — a caller-supplied tidy loadings frame → loading paths over a
  date/characteristic axis, or a loadings heatmap when no axis is given.
- :func:`plot_frontier` — a caller-supplied risk-return frontier frame → the efficient-frontier
  curve, optionally overlaid with named individual portfolios. :func:`mean_variance_frontier` is a
  small numpy-only convenience that traces such a frame from a mean vector and covariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numeraire.core.engine import PanelWeightsOutput, WeightsOutput
from plotnine import (
    aes,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    geom_tile,
    ggplot,
    labs,
)


def _weights_long(weights_output: WeightsOutput | PanelWeightsOutput) -> pd.DataFrame:
    """Flatten a weights Output to a tidy ``date, asset, weight`` frame (wide or long form)."""
    if isinstance(weights_output, WeightsOutput):
        wide = weights_output.weights
        long = wide.reset_index(names="date").melt(
            id_vars="date", var_name="asset", value_name="weight"
        )
    elif isinstance(weights_output, PanelWeightsOutput):
        s = weights_output.weights.rename("weight")
        long = s.reset_index()  # (date, asset) MultiIndex -> columns
    else:
        raise TypeError(
            "plot_weights_heatmap requires a WeightsOutput or PanelWeightsOutput; "
            f"got {type(weights_output).__name__}"
        )
    long["date"] = pd.to_datetime(long["date"])
    long["asset"] = long["asset"].astype(str)
    long["weight"] = long["weight"].astype(np.float64)
    return long[["date", "asset", "weight"]]


def plot_weights_heatmap(
    weights_output: WeightsOutput | PanelWeightsOutput,
    *,
    top: int | None = None,
    order: str = "mean",
) -> ggplot:
    """A date x asset portfolio-weight matrix as a heatmap, signed long/short about zero.

    Consumes a numeraire :class:`~numeraire.core.engine.WeightsOutput` (wide, fixed universe) or
    :class:`~numeraire.core.engine.PanelWeightsOutput` (long, entering/exiting universe) **object**
    directly — the weight stream is not in the tidy result schema, so this is a family-B plotter.
    Each ``(date, asset)`` weight is a ``geom_tile`` whose fill is the signed weight; compose with
    ``scale_fill_numeraire(diverging=True)`` so a long (positive) weight and a short (negative) one
    read as opposite hues about an unsaturated zero.

    ``top`` keeps only the ``N`` assets with the largest average *absolute* weight (the names the
    book actually leans on), dropping the long tail. ``order`` sorts the asset axis: ``"mean"``
    (default) orders by average weight so the persistent longs and shorts separate top-to-bottom,
    ``"name"`` orders alphabetically.
    """
    allowed = ("mean", "name")
    if order not in allowed:
        raise ValueError(f"order must be one of {allowed}; got {order!r}")
    long = _weights_long(weights_output)

    per_asset = long.groupby("asset")["weight"]
    mean_w = per_asset.mean()
    if top is not None:
        if top < 1:
            raise ValueError(f"top must be a positive integer; got {top}")
        keep = per_asset.apply(lambda w: w.abs().mean()).nlargest(top).index
        long = long[long["asset"].isin(keep)]
        mean_w = mean_w.loc[list(keep)]

    if order == "mean":
        assets = list(mean_w.sort_values().index)
    else:
        assets = sorted(mean_w.index, reverse=True)  # reversed so A reads at the top tile row
    long["asset"] = pd.Categorical(long["asset"], categories=assets, ordered=True)

    return (
        ggplot(long, aes(x="date", y="asset", fill="weight"))
        + geom_tile()
        + labs(x="", y="", fill="Weight")
    )


def _loadings_frame(loadings: pd.DataFrame) -> pd.DataFrame:
    """Validate a tidy loadings frame carries ``factor`` and ``loading``; return a typed copy."""
    if not isinstance(loadings, pd.DataFrame):
        raise TypeError(f"loadings must be a tidy DataFrame; got {type(loadings).__name__}")
    required = {"factor", "loading"}
    missing = required - {str(c) for c in loadings.columns}
    if missing:
        raise ValueError(
            f"loadings frame is missing required column(s) {sorted(missing)}; expected a tidy "
            "frame with 'factor', 'loading' and an axis such as 'date' or 'entity'"
        )
    out = loadings.copy()
    out["factor"] = out["factor"].astype(str)
    out["loading"] = out["loading"].astype(np.float64)
    return out


def plot_factor_loadings(loadings: pd.DataFrame, *, x: str | None = None) -> ggplot:
    """Factor-loading paths over an axis, or a loadings heatmap when no axis is given.

    There is **no** standard core loadings surface — a loadings object is method-local (an IPCA
    ``Gamma``, a rolling-beta panel), so this family-B plotter takes a caller-supplied tidy frame
    with columns ``factor``, ``loading`` and an axis (``date`` and/or ``entity``); a frame lacking
    ``factor`` or ``loading`` raises.

    With ``x`` given (a ``date`` for a time path, a characteristic column for a cross-sectional
    profile) the loadings are drawn as ``geom_line`` + ``geom_point`` paths coloured *and* facetted
    by factor. With ``x=None`` the frame is shown as a loadings heatmap — ``factor`` on the y-axis,
    the first present of ``entity``/``date`` on the x-axis, ``loading`` as the (diverging) fill —
    the natural view of a static factor x characteristic matrix.
    """
    data = _loadings_frame(loadings)

    if x is None:
        axis = next((c for c in ("entity", "date") if c in data.columns), None)
        if axis is None:
            raise ValueError(
                "a loadings heatmap needs an 'entity' or 'date' column for its x-axis; "
                "none found — pass x= to draw loading paths instead"
            )
        return (
            ggplot(data, aes(x=axis, y="factor", fill="loading"))
            + geom_tile()
            + labs(x="", y="", fill="Loading")
        )

    if x not in data.columns:
        raise ValueError(f"loading axis {x!r} is not a column of the loadings frame")
    data = data.sort_values(x, kind="stable")
    return (
        ggplot(data, aes(x=x, y="loading", color="factor"))
        + geom_hline(yintercept=0, color="#666666", size=0.3)
        + geom_line()
        + geom_point()
        + facet_wrap("~factor", scales="free_y")
        + labs(x=x, y="Loading", color="Factor")
    )


def _frontier_frame(frame: pd.DataFrame, what: str) -> pd.DataFrame:
    """Validate a risk-return frame carries ``risk`` and ``return``; return a typed copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    missing = {"risk", "return"} - {str(c) for c in frame.columns}
    if missing:
        raise ValueError(f"{what} frame is missing required column(s) {sorted(missing)}")
    out = frame.copy()
    out["risk"] = out["risk"].astype(np.float64)
    out["return"] = out["return"].astype(np.float64)
    return out


def plot_frontier(frontier: pd.DataFrame, *, points: pd.DataFrame | None = None) -> ggplot:
    """A mean-variance efficient frontier, optionally overlaid with named portfolios.

    Consumes a caller-supplied risk-return frame with columns ``risk`` and ``return`` (a frontier
    trace — from an optimizer, or from :func:`mean_variance_frontier`), drawn as the frontier curve
    (``geom_line`` + ``geom_point``). The frontier is not a framework result object, so this is a
    family-B plotter. ``points`` optionally overlays individual portfolios (``1/N``, GMV, tangency)
    as labelled markers: a frame with ``risk``, ``return`` and — for the labels — an optional
    ``label`` column. Both frames raise if they lack ``risk``/``return``.
    """
    front = _frontier_frame(frontier, "frontier").sort_values("risk", kind="stable")
    plot = (
        ggplot(front, aes(x="risk", y="return"))
        + geom_line()
        + geom_point()
        + labs(x="Risk (volatility)", y="Expected return")
    )
    if points is not None:
        pts = _frontier_frame(points, "points")
        plot = plot + geom_point(
            data=pts,
            mapping=aes(x="risk", y="return"),
            color="#D55E00",
            size=2.5,
            inherit_aes=False,
        )
        if "label" in pts.columns:
            plot = plot + geom_text(
                data=pts,
                mapping=aes(x="risk", y="return", label="label"),
                inherit_aes=False,
                va="bottom",
                ha="left",
            )
    return plot


def mean_variance_frontier(mean: np.ndarray, cov: np.ndarray, *, n: int = 50) -> pd.DataFrame:
    """Trace a risk-return efficient frontier from a mean vector and covariance (numpy only).

    A dependency-light convenience — *not* core, and not an optimizer — for callers who have a
    mean vector and covariance but no frontier frame yet. Uses the classic unconstrained (shorts
    allowed, no risk-free asset) closed form: over ``n`` target returns spanning the asset means,
    the minimum-variance portfolio's risk is ``sqrt((C*mu^2 - 2*A*mu + B) / D)`` with the standard
    efficient-set constants ``A, B, C`` and ``D = B*C - A^2``. Returns a ``risk``/``return`` frame
    ready for
    :func:`plot_frontier`.
    """
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sigma = np.asarray(cov, dtype=np.float64)
    k = mu.shape[0]
    if sigma.shape != (k, k):
        raise ValueError(
            f"cov must be square {k}x{k} to match mean of length {k}; got {sigma.shape}"
        )
    if n < 2:
        raise ValueError(f"n must be >= 2 to trace a frontier; got {n}")
    inv = np.linalg.inv(sigma)
    ones = np.ones(k, dtype=np.float64)
    a = float(ones @ inv @ mu)
    b = float(mu @ inv @ mu)
    c = float(ones @ inv @ ones)
    d = b * c - a * a
    if d <= 0.0:
        raise ValueError("degenerate frontier (B*C - A^2 <= 0); check the mean/covariance inputs")
    targets = np.linspace(float(mu.min()), float(mu.max()), n)
    variance = (c * targets**2 - 2.0 * a * targets + b) / d
    return pd.DataFrame({"risk": np.sqrt(variance), "return": targets})
