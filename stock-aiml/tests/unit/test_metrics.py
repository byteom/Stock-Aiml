"""Unit tests for metrics computation."""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from backend.services.metrics import compute_metrics, metrics_to_dict


class TestMetrics:
    def test_flat_equity_no_return(self):
        equity = [100_000.0] * 50
        m = compute_metrics(equity)
        assert abs(m.total_return) < 1e-6
        assert m.sharpe_ratio == 0.0
        assert m.max_drawdown == 0.0

    def test_positive_trending_returns(self):
        equity = [100_000 * (1.01 ** i) for i in range(100)]
        m = compute_metrics(equity)
        assert m.total_return > 0
        assert m.annualized_return > 0
        assert m.sharpe_ratio > 0

    def test_drawdown_is_reported(self):
        equity = [100_000, 110_000, 90_000, 95_000, 100_000]
        m = compute_metrics(equity)
        assert m.max_drawdown > 0
        assert m.max_drawdown_pct > 0
        assert m.max_drawdown_pct == abs(m.max_drawdown * 100)

    def test_metrics_to_dict_serialization(self):
        equity = list(range(100, 120))
        m = compute_metrics(equity)
        d = metrics_to_dict(m)
        assert isinstance(d, dict)
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d

    def test_cvar_is_negative_for_loss_series(self):
        # A series that always goes down
        equity = [100_000 / (1 + i * 0.02) for i in range(50)]
        m = compute_metrics(equity)
        assert m.cvar_95 <= 0

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_metrics([100_000.0])
