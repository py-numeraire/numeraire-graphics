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
from numeraire.core.engine import PanelWeightsOutput, WeightsOutput
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


@pytest.fixture
def ic_decay_results() -> pd.DataFrame:
    """``ic``-vs-horizon rows for two methods (the caller-joined ``horizon`` axis of plot_ic_decay).

    Shape mirrors what a caller assembles by running ``ICEvaluator`` on forecasts at several
    horizons and tagging each ``ic`` row with a numeric ``horizon`` — decaying with the horizon.
    """
    rows = []
    for method, base in (("voc", 0.09), ("hist_mean", 0.04)):
        for h, ic in zip([1, 3, 6, 12], [base, base * 0.6, base * 0.35, base * 0.1], strict=True):
            rows.append(
                _summary_row(method, "ic", ic, "n=100", capability="to_forecast", horizon=h)
            )
    return pd.DataFrame(rows)


@pytest.fixture
def weights_output() -> WeightsOutput:
    """A real long/short ``WeightsOutput`` (signed weights, 8 dates x 5 assets) for the heatmap."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2000-01-31", periods=8, freq="ME")
    assets = [f"a{i}" for i in range(5)]
    w = pd.DataFrame(rng.normal(0.0, 0.3, size=(8, 5)), index=dates, columns=assets)
    r = pd.DataFrame(rng.normal(0.0, 0.04, size=(8, 5)), index=dates, columns=assets)
    return WeightsOutput(
        weights=w, realized=r, method="ls", config_hash="cfg0", data_vintage="synthetic", run_id="r"
    )


@pytest.fixture
def panel_weights_output() -> PanelWeightsOutput:
    """A real ``PanelWeightsOutput`` over a ragged (entering/exiting) long-short panel.

    Long ``(date, asset)`` weights on a MultiIndex — the shape the cross-sectional engine emits;
    the universe changes across dates, which the wide ``WeightsOutput`` cannot represent.
    """
    rng = np.random.default_rng(11)
    dates = pd.date_range("2000-01-31", periods=4, freq="ME")
    keys: list[tuple[pd.Timestamp, str]] = []
    for i, t in enumerate(dates):
        # ragged: the universe shifts by one name each date (a2..a5 window)
        for a in range(i, i + 4):
            keys.append((t, f"a{a}"))
    idx = pd.MultiIndex.from_tuples(keys, names=["date", "asset"])
    w = pd.Series(rng.normal(0.0, 0.3, size=len(idx)), index=idx, name="weight")
    r = pd.Series(rng.normal(0.0, 0.04, size=len(idx)), index=idx, name="realized")
    return PanelWeightsOutput(
        weights=w, realized=r, method="ls", config_hash="cfg0", data_vintage="synthetic", run_id="r"
    )
