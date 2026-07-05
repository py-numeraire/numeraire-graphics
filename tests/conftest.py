"""Synthetic, schema-conformant result frames for the figure tests.

Two builders: ``strategy_return_results`` runs a hand-built ``WeightsOutput`` through numeraire's
real per-date and summary evaluators (so the time-series plots are tested against the genuine data
contract, not a mock), and ``summary_results`` hand-crafts scalar rows for the bar / curve plots.
All data is synthetic — no external or licensed data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numeraire.core.engine import WeightsOutput
from numeraire.core.evaluators import (
    MeanReturnEvaluator,
    SharpeEvaluator,
    StrategyReturnEvaluator,
)
from numeraire.core.schema import RESULT_COLUMNS


def _weights_output(method: str, seed: int, n_dates: int = 60, n_assets: int = 4) -> WeightsOutput:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=n_dates, freq="ME")
    assets = [f"a{i}" for i in range(n_assets)]
    w = np.full((n_dates, n_assets), 1.0 / n_assets)
    r = rng.normal(0.008, 0.04, size=(n_dates, n_assets))
    return WeightsOutput(
        weights=pd.DataFrame(w, index=dates, columns=assets),
        realized=pd.DataFrame(r, index=dates, columns=assets),
        method=method,
        config_hash="cfg0",
        data_vintage="synthetic",
        run_id=f"{method}-run",
    )


@pytest.fixture
def strategy_return_results() -> pd.DataFrame:
    """Per-date ``strategy_return`` rows plus ``sharpe``/``mean_return`` summaries, two methods."""
    per_date = StrategyReturnEvaluator()
    sharpe = SharpeEvaluator()
    mean = MeanReturnEvaluator()
    frames: list[pd.DataFrame] = []
    for method, seed in (("model_a", 1), ("model_b", 2)):
        out = _weights_output(method, seed)
        frames += [per_date.evaluate(out), sharpe.evaluate(out), mean.evaluate(out)]
    return pd.concat(frames, ignore_index=True)


def _summary_row(method: str, metric: str, value: float, universe: str, **extra: object) -> dict:
    row = {
        "run_id": f"{method}-run",
        "method": method,
        "date": pd.Timestamp("2010-12-31"),
        "metric": metric,
        "value": value,
        "universe": universe,
        "capability": "to_weights",
        "protocol": "walk_forward",
        "config_hash": "cfg0",
        "data_vintage": "synthetic",
    }
    row.update(extra)
    return row


@pytest.fixture
def summary_results() -> pd.DataFrame:
    """Scalar ``sharpe`` rows across methods and universes (repeated rows enable a derived CI)."""
    rows = [
        _summary_row("model_a", "sharpe", 0.9, "n=25"),
        _summary_row("model_a", "sharpe", 1.1, "n=30"),
        _summary_row("model_b", "sharpe", 0.4, "n=25"),
        _summary_row("model_b", "sharpe", 0.6, "n=30"),
    ]
    return pd.DataFrame(rows, columns=[*RESULT_COLUMNS])


@pytest.fixture
def complexity_results() -> pd.DataFrame:
    """A ``sharpe``-vs-complexity sweep for one method, with ribbon (``lo``/``hi``) columns."""
    rows = []
    for c, val in zip([1, 2, 4, 8, 16], [0.2, 0.5, 0.7, 0.65, 0.55], strict=True):
        rows.append(
            _summary_row("voc", "sharpe", val, "n=100", complexity=c, lo=val - 0.1, hi=val + 0.1)
        )
    return pd.DataFrame(rows)
