"""Tests for the tidy-schema access helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from numeraire_viz._common import require_columns, series_rows, summary_rows


def test_require_columns_rejects_non_schema_frame():
    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(pd.DataFrame({"foo": [1]}))


def test_series_rows_filters_and_sorts(strategy_return_results):
    sub = series_rows(strategy_return_results, "strategy_return")
    assert (sub["metric"] == "strategy_return").all()
    assert pd.api.types.is_datetime64_any_dtype(sub["date"])
    # sorted within method
    for _, grp in sub.groupby("method"):
        assert list(grp["date"]) == sorted(grp["date"])


def test_series_rows_missing_metric_raises(strategy_return_results):
    with pytest.raises(ValueError, match="no rows with metric"):
        series_rows(strategy_return_results, "not_a_metric")


def test_summary_rows_filters(strategy_return_results):
    sub = summary_rows(strategy_return_results, "sharpe")
    assert (sub["metric"] == "sharpe").all()
    assert len(sub) == 2  # one per method


def test_summary_rows_missing_metric_raises(strategy_return_results):
    with pytest.raises(ValueError, match="no rows with metric"):
        summary_rows(strategy_return_results, "nope")
