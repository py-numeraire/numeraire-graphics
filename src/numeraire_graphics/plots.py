"""The result-schema-driven figures.

Four builders, one contract: each takes a tidy result DataFrame as the first positional argument,
keyword-only options after, and **returns a plotnine grammar object** (never draws or saves). The
caller composes freely (``plot + theme_numeraire() + scale_color_numeraire()``) and, when ready,
hands the object to :func:`numeraire_graphics.save_paper`.

Schema mapping in one line each:

- :func:`plot_cumulative` — ``metric == "strategy_return"`` per-date rows → wealth + drawdown.
- :func:`plot_rolling` — the same per-date returns → a rolling statistic (default Sharpe).
- :func:`plot_metric_by` — a scalar summary metric → bars (with CI whiskers when derivable).
- :func:`plot_complexity_curve` — a scalar metric against a caller-supplied numeric axis → a curve.
- :func:`plot_ic_decay` — the ``ic`` rows against a caller-supplied numeric ``horizon`` → a curve.

These are the **result-schema plotters** (input family A). The richer object/frame plotters that
need inputs the tidy schema does not carry — a weight stream, a loadings panel, a frontier trace —
live in :mod:`numeraire_graphics.outputs` (input family B).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    facet_wrap,
    geom_col,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    geom_rect,
    geom_ribbon,
    geom_smooth,
    ggplot,
    labs,
    scale_y_continuous,
)

from numeraire_graphics._common import (
    series_rows,
    smart_date_scale,
    summary_rows,
    warn_palette_overflow,
)


def _percent_labels() -> Any:
    """A percent-axis label callable, tolerant of the mizani label-API rename."""
    try:
        from mizani.labels import label_percent

        return label_percent()
    except ImportError:  # older mizani
        from mizani.formatters import percent_format

        return percent_format()


def _wealth_and_drawdown(returns: pd.Series) -> pd.DataFrame:
    """Cumulative (geometric) return and drawdown for one date-ordered return series."""
    r = returns.to_numpy(dtype=np.float64)
    if not np.isfinite(r).all():
        raise ValueError("strategy returns must be finite; missing returns are not zero returns")
    if (r <= -1.0).any():
        raise ValueError("simple strategy returns must be greater than -1 for geometric wealth")
    wealth = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(wealth)
    return pd.DataFrame(
        {
            "date": returns.index,
            "Cumulative return": wealth - 1.0,
            "Drawdown": wealth / running_max - 1.0,
        }
    )


def _recession_frame(recessions: Any) -> pd.DataFrame:
    """Normalize a recessions argument to a frame with datetime ``start``/``end`` columns."""
    if isinstance(recessions, pd.DataFrame):
        if not {"start", "end"} <= set(recessions.columns):
            raise ValueError("recessions frame must have 'start' and 'end' columns")
        pairs = recessions[["start", "end"]].to_numpy().tolist()
    else:
        pairs = [list(p) for p in recessions]
    starts = pd.to_datetime([p[0] for p in pairs])
    ends = pd.to_datetime([p[1] for p in pairs])
    return pd.DataFrame({"start": starts, "end": ends})


def plot_cumulative(
    results: pd.DataFrame,
    *,
    benchmark: str | pd.Series | None = None,
    recessions: Any = None,
) -> ggplot:
    """Cumulative-return and drawdown curves from per-date strategy-return rows.

    Consumes the ``metric == "strategy_return"`` rows (one per date per method, as emitted by
    ``StrategyReturnEvaluator``): each method's returns are geometrically compounded into a wealth
    curve and its drawdown, shown in stacked facets (``geom_line`` coloured by method, a zero
    reference line).

    ``benchmark`` overlays a reference as a distinct dashed line: pass a *method name* already in
    ``results`` (it is drawn as the benchmark rather than a peer) or a date-indexed ``pd.Series`` of
    returns supplied by the caller. ``recessions`` shades caller-supplied ``(start, end)`` date
    spans via ``geom_rect`` — the caller provides the dates; nothing is fetched.
    """
    series = series_rows(results, "strategy_return")
    calendars = [tuple(group["date"]) for _, group in series.groupby("method", sort=False)]
    if not calendars:
        raise ValueError(
            "no strategy-return series to plot; the result table has no usable per-date rows"
        )
    if any(calendar != calendars[0] for calendar in calendars[1:]):
        raise ValueError(
            "cumulative comparisons require exactly aligned method calendars; align the result "
            "table explicitly before plotting"
        )

    panels: list[pd.DataFrame] = []
    bench_name: str | None = None
    if isinstance(benchmark, str):
        present = list(pd.unique(series["method"]))
        if benchmark not in present:
            raise ValueError(
                f"benchmark={benchmark!r} is not a method in the result table; present methods are "
                f"{present}. Pass one of those names, or a date-indexed pd.Series of returns."
            )
        bench_name = benchmark
    for method, grp in series.groupby("method", sort=False):
        ret = pd.Series(grp["value"].to_numpy(dtype=np.float64), index=grp["date"])
        wd = _wealth_and_drawdown(ret)
        wd["method"] = str(method)
        wd["role"] = "benchmark" if str(method) == bench_name else "strategy"
        panels.append(wd)
    if isinstance(benchmark, pd.Series):
        ret = pd.Series(benchmark.to_numpy(dtype=np.float64), index=pd.to_datetime(benchmark.index))
        if not ret.index.is_unique:
            raise ValueError("benchmark index must be unique")
        ret = ret.sort_index()
        if tuple(ret.index) != calendars[0]:
            raise ValueError(
                "benchmark dates must exactly match the strategy calendar; align it explicitly "
                "before plotting"
            )
        wd = _wealth_and_drawdown(ret)
        wd["method"] = str(benchmark.name) if benchmark.name is not None else "benchmark"
        wd["role"] = "benchmark"
        panels.append(wd)

    wide = pd.concat(panels, ignore_index=True)
    warn_palette_overflow(wide["method"].nunique())
    long = wide.melt(
        id_vars=["date", "method", "role"],
        value_vars=["Cumulative return", "Drawdown"],
        var_name="panel",
        value_name="value",
    )

    plot = ggplot(long, aes(x="date", y="value"))
    if recessions is not None:
        rec = _recession_frame(recessions)
        plot = plot + geom_rect(
            mapping=aes(xmin="start", xmax="end"),
            data=rec,
            ymin=-np.inf,
            ymax=np.inf,
            fill="#999999",
            alpha=0.20,
            inherit_aes=False,
        )
    plot = (
        plot
        + geom_hline(yintercept=0, color="#666666", size=0.3)
        + geom_line(aes(color="method", linetype="role"))
        + facet_wrap("~panel", ncol=1, scales="free_y")
        + smart_date_scale(long["date"])
        + scale_y_continuous(labels=_percent_labels())
        + labs(x="", y="", color="Method", linetype="Role")
    )
    return plot


def plot_rolling(
    results: pd.DataFrame,
    *,
    window: int,
    metric: str = "sharpe",
) -> ggplot:
    """A rolling statistic of the per-date strategy returns (default rolling Sharpe).

    Reads the ``metric == "strategy_return"`` rows and, per method, computes a trailing ``window``
    statistic: ``"sharpe"`` (mean / sample std over the window), ``"mean"`` (mean return), or
    ``"vol"`` (sample std). Returns a ``geom_line`` coloured by method with a zero reference; the
    y-axis is a percent scale for the return-unit statistics, plain for the (unitless) Sharpe.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    allowed = ("sharpe", "mean", "vol")
    if metric not in allowed:
        raise ValueError(f"metric must be one of {allowed}; got {metric!r}")
    series = series_rows(results, "strategy_return")
    calendars = [tuple(group["date"]) for _, group in series.groupby("method", sort=False)]
    if not calendars:
        raise ValueError(
            "no strategy-return series to plot; the result table has no usable per-date rows"
        )
    if any(calendar != calendars[0] for calendar in calendars[1:]):
        raise ValueError(
            "rolling comparisons require exactly aligned method calendars; align the result "
            "table explicitly before plotting"
        )

    frames: list[pd.DataFrame] = []
    for method, grp in series.groupby("method", sort=False):
        ret = pd.Series(grp["value"].to_numpy(dtype=np.float64), index=grp["date"])
        roll = ret.rolling(window)
        if metric == "sharpe":
            stat = roll.mean() / roll.std(ddof=1)
        elif metric == "mean":
            stat = roll.mean()
        else:
            stat = roll.std(ddof=1)
        out = pd.DataFrame({"date": ret.index, "value": stat.to_numpy()})
        out["method"] = str(method)
        frames.append(out.dropna(subset=["value"]))
    data = pd.concat(frames, ignore_index=True)
    warn_palette_overflow(data["method"].nunique())

    ylab = {"sharpe": f"Rolling Sharpe ({window})", "mean": "Rolling mean", "vol": "Rolling vol"}[
        metric
    ]
    plot = (
        ggplot(data, aes(x="date", y="value", color="method"))
        + geom_hline(yintercept=0, color="#666666", size=0.3)
        + geom_line()
        + smart_date_scale(data["date"])
        + labs(x="", y=ylab, color="Method")
    )
    if metric != "sharpe":
        plot = plot + scale_y_continuous(labels=_percent_labels())
    return plot


def _derive_ci(sub: pd.DataFrame, x: str) -> pd.DataFrame:
    """Collapse summary rows to one ``value`` per ``x`` group, adding ``lo``/``hi`` when derivable.

    A confidence band is taken from (in order): explicit ``ci_low``/``ci_high`` columns; a standard-
    error column ``se`` (±1.96 se). Repeated rows are not treated as IID replications: they often
    differ by universe, vintage, configuration, or sample and must be aggregated upstream with a
    declared estimand and uncertainty procedure.
    """
    has_ci = {"ci_low", "ci_high"} <= set(sub.columns)
    has_se = "se" in sub.columns
    records: list[dict[str, Any]] = []
    for key, grp in sub.groupby(x, sort=False):
        if len(grp) != 1:
            raise ValueError(
                f"metric={grp['metric'].iloc[0]!r} has {len(grp)} rows for {x}={key!r}; "
                "filter or aggregate them explicitly instead of treating heterogeneous rows as "
                "IID replications"
            )
        value = float(grp["value"].iloc[0])
        lo = hi = np.nan
        if has_ci:
            lo, hi = float(grp["ci_low"].iloc[0]), float(grp["ci_high"].iloc[0])
        elif has_se:
            se = float(grp["se"].iloc[0])
            lo, hi = value - 1.96 * se, value + 1.96 * se
        records.append({x: key, "value": value, "lo": lo, "hi": hi})
    return pd.DataFrame.from_records(records)


def plot_metric_by(
    results: pd.DataFrame,
    *,
    metric: str,
    x: str = "method",
) -> ggplot:
    """A bar chart of a summary ``metric`` across a grouping column ``x`` (method, universe, ...).

    Filters ``results`` to the scalar ``metric`` (one row per group, e.g. ``"sharpe"``) and draws
    ``geom_col``. When a confidence interval is declared through explicit ``ci_low``/``ci_high``
    or a standard-error column ``se``, it is added as ``geom_errorbar`` whiskers; otherwise the
    bars stand plain. Repeated rows for one ``x`` value raise because universes, vintages, and runs
    are not IID replicates. A zero reference line anchors signed metrics.
    """
    sub = summary_rows(results, metric)
    if x not in sub.columns:
        raise ValueError(f"grouping column {x!r} is not in the result table")
    data = _derive_ci(sub, x)
    has_ci = bool(data["hi"].notna().any())
    warn_palette_overflow(len(data))

    plot = (
        ggplot(data, aes(x=x, y="value", fill=x))
        + geom_hline(yintercept=0, color="#666666", size=0.3)
        # ``fill`` encodes the same column as ``x``, so a fill legend would just restate the axis
        # tick labels — suppress it (the coloured bars remain, keyed by the x-axis).
        + geom_col(show_legend=False)
    )
    if has_ci:
        plot = plot + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.25)
    return plot + labs(x=x, y=metric)


def plot_complexity_curve(
    results: pd.DataFrame,
    *,
    x: str,
    metric: str,
    ribbon: tuple[str, str] | None = None,
) -> ggplot:
    """A summary ``metric`` plotted against a numeric complexity axis ``x``.

    ``x`` names a numeric column the caller has joined onto ``results`` (a shrinkage intensity, a
    parameter count, a regularization level — the result schema does not carry one, so it is an
    explicit argument). Rows are sorted along ``x`` and drawn as a ``geom_line`` + ``geom_point``
    coloured by method. ``ribbon`` optionally names ``(low, high)`` columns for a ``geom_ribbon``
    band around the curve. A zero reference line is included.
    """
    sub = summary_rows(results, metric)
    if x not in sub.columns:
        raise ValueError(
            f"complexity axis {x!r} is not in the result table; join it on before plotting"
        )
    data = sub.sort_values(x, kind="stable")
    warn_palette_overflow(data["method"].nunique())

    plot = ggplot(data, aes(x=x, y="value", color="method")) + geom_hline(
        yintercept=0, color="#666666", size=0.3
    )
    if ribbon is not None:
        low, high = ribbon
        for col in (low, high):
            if col not in data.columns:
                raise ValueError(f"ribbon column {col!r} is not in the result table")
        plot = plot + geom_ribbon(aes(ymin=low, ymax=high, fill="method"), alpha=0.20, color=None)
    plot = plot + geom_line() + geom_point() + labs(x=x, y=metric, color="Method")
    return plot


def plot_ic_decay(
    results: pd.DataFrame,
    *,
    horizon: str = "horizon",
    metric: str = "ic",
    smooth: bool = False,
) -> ggplot:
    """The information coefficient plotted against a caller-assembled forecast-horizon axis.

    Consumes the ``ic`` rows :class:`ICEvaluator` emits (``metric="ic"`` by default; ``"ic_ir"``
    or ``"ic_t"`` read the same way). A single :class:`ForecastOutput` carries one horizon, so its
    ``ic`` row is scalar; the *decay curve* is assembled by the caller running forecasts at several
    horizons and tagging each ``ic`` row with a numeric ``horizon`` column, then stacking them. The
    result schema has no horizon column — exactly the caller-supplied-axis pattern of
    :func:`plot_complexity_curve` — so ``horizon`` is an explicit argument and its absence raises.

    Rows are sorted along ``horizon`` and drawn as a ``geom_line`` + ``geom_point`` coloured by
    method, over a zero reference line (an IC decaying toward zero as the horizon lengthens is the
    figure's whole point). ``smooth=True`` overlays a light linear trend (``geom_smooth``) per
    method for the eye.
    """
    sub = summary_rows(results, metric)
    if horizon not in sub.columns:
        raise ValueError(
            f"horizon axis {horizon!r} is not in the result table; the caller assembles it by "
            "running forecasts at several horizons and tagging each 'ic' row, then joining it on"
        )
    data = sub.sort_values(horizon, kind="stable")
    warn_palette_overflow(data["method"].nunique())

    plot = ggplot(data, aes(x=horizon, y="value", color="method")) + geom_hline(
        yintercept=0, color="#666666", size=0.3
    )
    if smooth:
        plot = plot + geom_smooth(method="lm", se=False, linetype="dashed")
    plot = plot + geom_line() + geom_point() + labs(x=horizon, y=metric, color="Method")
    return plot
