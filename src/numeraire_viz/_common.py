"""Shared internals: the colourblind palette and small tidy-schema access helpers.

Every public plot consumes the numeraire tidy result schema
(``run_id, method, date, metric, value, universe, capability, protocol, config_hash,
data_vintage``). The helpers here isolate the two schema idioms the figures rely on: the
per-date *series* rows a time-indexed evaluator emits (one row per date per method) and the
single *summary* rows a scalar evaluator emits (one row per method). Nothing here draws.
"""

from __future__ import annotations

import pandas as pd
from numeraire.core.schema import validate_result

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
