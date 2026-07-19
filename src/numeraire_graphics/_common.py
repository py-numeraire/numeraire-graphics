"""Shared internals: the colourblind palette and small tidy-schema access helpers.

Every public plot consumes the numeraire tidy result schema
(``run_id, method, date, metric, value, universe, capability, protocol, config_hash,
data_vintage``). The helpers here isolate the two schema idioms the figures rely on: the
per-date *series* rows a time-indexed evaluator emits (one row per date per method) and the
single *summary* rows a scalar evaluator emits (one row per method). Nothing here draws.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from numeraire.core.schema import validate_result
from plotnine import scale_x_datetime

# Okabe & Ito (2008) qualitative palette — the eight-colour set that stays distinguishable under
# the common forms of colour-vision deficiency. Ordered so the first draws are the most separable.
OKABE_ITO: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

# Greyscale-safe line styles, cycled alongside colour so figures survive a monochrome print.
GREYSCALE_LINETYPES: tuple[str, ...] = ("solid", "dashed", "dotted", "dashdot")


def require_columns(results: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``results`` is not a conforming tidy result table.

    Thin re-export of numeraire's own schema check so callers get one consistent error.
    """
    validate_result(results)


def series_rows(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return the per-date rows for a time-indexed ``metric`` (e.g. ``"strategy_return"``).

    Rows are filtered to the metric, the ``date`` column is parsed to datetimes, and the frame
    is sorted by ``(method, date)`` so downstream cumulation/rolling is well defined. Raises
    ``ValueError`` if the metric is absent.
    """
    require_columns(results)
    sub = results.loc[results["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(
            f"no rows with metric={metric!r}; a time-indexed evaluator must run first "
            "(e.g. StrategyReturnEvaluator emits metric='strategy_return')"
        )
    sub["date"] = pd.to_datetime(sub["date"])
    values = pd.to_numeric(sub["value"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            f"metric={metric!r} contains non-finite values; refuse to plot gaps as data"
        )
    duplicate = sub.duplicated(["method", "date"], keep=False)
    if duplicate.any():
        methods = sorted(str(item) for item in sub.loc[duplicate, "method"].unique())
        raise ValueError(
            "per-date plots require one observation per (method, date); duplicate rows for "
            f"{methods} may indicate overlapping folds or mixed runs"
        )
    identity = [
        "run_id",
        "universe",
        "capability",
        "protocol",
        "config_hash",
        "data_vintage",
    ]
    mixed = []
    for method, group in sub.groupby("method", sort=False):
        if any(group[column].nunique(dropna=False) != 1 for column in identity):
            mixed.append(str(method))
    if mixed:
        raise ValueError(
            "per-date plots cannot silently combine multiple runs/configurations under one "
            f"method label: {mixed}; filter the result table or give each curve a unique method"
        )
    return sub.sort_values(["method", "date"], kind="stable").reset_index(drop=True)


def summary_rows(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return the rows for a scalar summary ``metric`` (e.g. ``"sharpe"``), schema-validated.

    No date parsing (a summary row's ``date`` is the sample's last date, incidental here).
    Raises ``ValueError`` if the metric is absent.
    """
    require_columns(results)
    sub = results.loc[results["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"no rows with metric={metric!r} in the result table")
    return sub.reset_index(drop=True)


# --- date-axis formatting ------------------------------------------------------------------------
# The default plotnine datetime axis places full-ISO labels ("2012-01-01") at unbounded breaks, so
# a multi-year daily/monthly series crowds them into an unreadable run. These helpers mirror the
# span-adaptive behaviour of matplotlib's ConciseDateFormatter + AutoDateLocator: pick a break width
# and a compact strftime label from the *span* of the series, capping the break count so labels
# never overlap. Applied inside every datetime-x plot builder (not left to the caller).

# Target number of x-axis breaks — a readable band on a single-column figure. mizani snaps a "N
# years"/"N months" width to calendar boundaries, so the realised count lands within ~1 of this.
_DATE_BREAK_TARGET = 7
_DAYS_PER_YEAR = 365.25
_DAYS_PER_MONTH = 30.44


def date_breaks_and_labels(dates: Any) -> tuple[str, str]:
    """Choose a ``(date_breaks, date_labels)`` pair for a datetime axis from the series' span.

    Returns a mizani break-width string (``"3 years"``, ``"2 months"``, ``"5 days"``) and a compact
    strftime format: year-only (``"%Y"``) for multi-year spans, month-and-year (``"%b %Y"``) for
    ~4 months to a few years, day-and-month (``"%d %b"``) for a short span. The break width is
    scaled so about :data:`_DATE_BREAK_TARGET` labels appear regardless of span, never overlapping.
    """
    dts = pd.to_datetime(pd.Series(list(dates))).dropna()
    if dts.empty:
        return "1 year", "%Y"
    span_days = max(int((dts.max() - dts.min()).days), 1)
    years = span_days / _DAYS_PER_YEAR
    if years >= 2.5:
        step = max(1, math.ceil(years / _DATE_BREAK_TARGET))
        return f"{step} years", "%Y"
    if span_days >= 120:  # ~4 months up to ~2.5 years → month granularity
        months = max(1, round(span_days / _DAYS_PER_MONTH))
        step = max(1, math.ceil(months / _DATE_BREAK_TARGET))
        return f"{step} months", "%b %Y"
    step = max(1, math.ceil(span_days / _DATE_BREAK_TARGET))
    return f"{step} days", "%d %b"


def smart_date_scale(dates: Any) -> scale_x_datetime:
    """A ``scale_x_datetime`` with span-adaptive, non-overlapping breaks and compact labels."""
    breaks, labels = date_breaks_and_labels(dates)
    return scale_x_datetime(date_breaks=breaks, date_labels=labels)


def thinned_break_labels(labels: list[str], *, target: int = 12) -> list[str]:
    """Keep an evenly-spaced subset (~``target``) of ordered category labels for a discrete axis.

    The weight heatmap treats its rebalance-date axis as an ordered factor (so tiles abut with no
    gaps); labelling *every* date would crowd the axis, so only every k-th label is shown.
    """
    if len(labels) <= target:
        return list(labels)
    step = math.ceil(len(labels) / target)
    return list(labels[::step])


def warn_palette_overflow(n_groups: int) -> None:
    """Warn when a comparison has more groups than the 8-colour Okabe-Ito palette can distinguish.

    Beyond eight the discrete scales cycle, so colours repeat; the caller should facet, thin the
    comparison, or switch to ``greyscale=True`` with varied linetypes for separability.
    """
    if n_groups > len(OKABE_ITO):
        warnings.warn(
            f"{n_groups} methods exceed the {len(OKABE_ITO)}-colour Okabe-Ito palette; colours "
            "will cycle and repeat — facet, thin the comparison, or use greyscale + linetypes.",
            stacklevel=3,
        )
