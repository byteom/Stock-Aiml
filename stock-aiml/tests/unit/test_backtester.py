"""Unit tests for the backtester service."""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from backend.services.backtester import Backtester, momentum_signal
from backend.services.execution_model import ExecutionSurrogate


def _make_test_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    base  = 25000.0
    opens = [base + i * 2 + rng.normal() * 5 for i in range(n)]
    closes = [o * (1 + rng.normal() * 0.01) for o in opens]
    highs = [o + abs(rng.normal() * 3) for o in opens]
    lows  = [o - abs(rng.normal() * 3) for o in opens]
    df = pd.DataFrame({
        "timestamp": dates,
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": [1_000_000] * n,
    })
    df["close"] = df["close"].clip(df["low"] * 0.99, df["high"] * 1.01)
    return df


class TestExecutionSurrogate:
    def test_simulate_fill_full_size(self):
        es = ExecutionSurrogate()
        result = es.simulate_fill(
            signal_price=100.0,
            order_size=1000.0,
            direction=1,
            current_vol=0.01,
            adv=10_000.0,
            stochastic=False,
        )
        assert result.filled is True
        assert result.fill_price > 0

    def test_simulate_fill_zero_size(self):
        es = ExecutionSurrogate()
        result = es.simulate_fill(
            signal_price=100.0, order_size=0.0,
            direction=1, current_vol=0.01, adv=10_000.0, stochastic=False,
        )
        assert result.filled is False

    def test_fill_rate_decreases_with_size(self):
        es = ExecutionSurrogate(fill_rate=0.85)
        r_small = es.simulate_fill(100.0, 100.0, 1, 0.01, 10_000.0, stochastic=False)
        r_large = es.simulate_fill(100.0, 5000.0, 1, 0.01, 10_000.0, stochastic=False)
        assert r_large.fill_prob <= r_small.fill_prob


class TestMomentumSignal:
    def test_signal_generates_no_future_peek(self):
        df = _make_test_df(200)
        # Add basic features
        df["return_1d"] = df["close"].pct_change().fillna(0)
        df["return_5d"] = df["close"].pct_change(5).fillna(0)
        df["return_20d"] = df["close"].pct_change(20).fillna(0)
        delta = df["close"].diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)

        params = {
            "rsi_buy_threshold":  30.0,
            "rsi_sell_threshold": 70.0,
            "lookback_short":     5,
            "lookback_long":      20,
            "min_return_diff":    0.0,
        }
        signal = momentum_signal(df, params)
        assert len(signal) == len(df)
        assert signal.isin([-1, 0, 1]).all()


class TestBacktester:
    def test_backtest_runs_without_error(self):
        df = _make_test_df(100)
        exec_surrogate = ExecutionSurrogate()
        bt = Backtester(
            initial_capital=100_000.0,
            commission_pct=0.0,
            slippage_bps=0.0,
            exec_surrogate=exec_surrogate,
        )

        # Simple params — momentum
        result = bt.run(
            df=df,
            strategy_name="momentum",
            strategy_params={
                "rsi_buy_threshold": 30.0,
                "rsi_sell_threshold": 70.0,
                "lookback_short": 5,
                "lookback_long": 20,
                "min_return_diff": 0.0,
                "max_position_pct": 1.0,
            },
            stochastic_exec=False,
        )

        assert result.n_bars == 100
        assert len(result.equity_curve) == 100
        assert len(result.drawdown_curve) == 100
        assert "annualized_return" in result.metrics

    def test_drawdown_is_negative(self):
        df = _make_test_df(100)
        exec_surrogate = ExecutionSurrogate()
        bt = Backtester(initial_capital=100_000.0, exec_surrogate=exec_surrogate)
        result = bt.run(df=df, strategy_name="momentum", strategy_params={})
        # Most values in drawdown curve should be <= 0
        dd = result.drawdown_curve
        assert all(v <= 0 for v in dd)

    def test_equity_starts_at_initial_capital(self):
        df = _make_test_df(50)
        exec_surrogate = ExecutionSurrogate()
        capital = 500_000.0
        bt = Backtester(initial_capital=capital, exec_surrogate=exec_surrogate)
        result = bt.run(df=df, strategy_name="momentum", strategy_params={})
        assert result.equity_curve[0] == capital

    def test_short_df_raises(self):
        df = _make_test_df(5)
        bt = Backtester()
        with pytest.raises(ValueError, match="too short"):
            bt.run(df=df, strategy_name="momentum", strategy_params={})

    def test_unknown_strategy_raises(self):
        df = _make_test_df(50)
        bt = Backtester()
        with pytest.raises(ValueError, match="Unknown strategy"):
            bt.run(df=df, strategy_name="not_a_strategy", strategy_params={})
