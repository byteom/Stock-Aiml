"""Execution-aware backtesting engine.

This is the core backtester that:
1. Reads a strategy configuration (signal generation rules)
2. Simulates order execution using the ExecutionSurrogate
3. Applies transaction costs (commission + slippage)
4. Tracks position, equity, and generates trade-level logs
5. Computes performance metrics

No lookahead bias — all signal computation uses only past + current bar data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from backend.services.execution_model import ExecutionSurrogate, FillResult
from backend.services.metrics import compute_metrics, metrics_to_dict


# ─── Enums & dataclasses ───────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    """A single executed trade (entry + exit pair)."""
    entry_date:  str
    exit_date:   str
    side:        str          # "buy" | "sell"
    entry_price: float
    exit_price:  float
    size:        float
    pnl:         float
    return_pct:  float
    hold_bars:   int
    notional:    float
    commission: float
    slippage:    float
    fill_prob:   float


@dataclass
class BacktestState:
    """Mutable per-bar state during a backtest run."""
    capital:      float
    position:     float       # shares/units held (+ long, - short)
    position_avg: float      # average entry price
    entry_bar:    int        # bar index when position was entered
    pending_order: dict | None  # waiting-bar order for confirmation
    prev_close:   float = 0.0  # previous bar close price (for equity continuity)


@dataclass
class BacktestResult:
    """Full result of a backtest run."""
    strategy_name:     str
    equity_curve:      list[float]
    timestamps:       list[str]
    trades:           list[dict]
    metrics:          dict
    drawdown_curve:   list[float]
    config:            dict
    split_period:     str
    n_bars:           int
    trade_log:        list[dict] = field(default_factory=list)


# ─── Strategy signal generators ───────────────────────────────────────────────

def momentum_signal(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Momentum strategy signal:
      +1 (bullish) when long return > short return (uptrend confirmed)
      -1 (bearish) when long return < short return (downtrend confirmed)
       0 (neutral) otherwise

    Convention: spread = long_return - short_return
    Positive spread = price above long-term trend = bullish
    """
    lookback_short  = params.get("lookback_short", 5)
    lookback_long   = params.get("lookback_long",  20)
    min_spread      = params.get("min_return_diff", 0.0)

    ret_s = df.get(f"return_{lookback_short}d", pd.Series(0, index=df.index))
    ret_l = df.get(f"return_{lookback_long}d", pd.Series(0, index=df.index))

    # spread = long - short (positive when price above long-term avg)
    spread = ret_l - ret_s
    bullish = spread > min_spread
    bearish = spread < -min_spread

    signal = pd.Series(0, index=df.index)
    signal[bullish] =  1
    signal[bearish] = -1
    return signal


def mean_reversion_signal(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Mean reversion strategy signal:
      +1 (buy oversold) when z-score < z_entry_threshold
      -1 (sell overbought) when z-score > z_exit_threshold
       0 otherwise
    """
    z_entry = params.get("z_entry_threshold", -2.0)
    z_exit  = params.get("z_exit_threshold",   0.5)
    bb_pos  = df.get("bb_position", df.get("bb_pos", pd.Series(0.5, index=df.index)))

    signal = pd.Series(0, index=df.index)
    signal[bb_pos < z_entry] =  1   # oversold -> buy
    signal[bb_pos > z_exit]  = -1   # overbought -> sell
    return signal


STRATEGY_SIGNALS = {
    "momentum":       momentum_signal,
    "mean_reversion": mean_reversion_signal,
}


# ─── Main backtester ────────────────────────────────────────────────────────────

class Backtester:
    """
    Execution-aware backtester.

    Workflow per bar:
      1. Compute signal using strategy function (NO future data)
      2. Generate order if signal flipped vs current position
      3. Simulate execution via ExecutionSurrogate
      4. Update position and capital
      5. Check stop-loss / take-profit / time exit conditions
      6. Record equity
    """

    def __init__(
        self,
        initial_capital:  float = 1_000_000.0,
        commission_pct:  float = 0.001,
        slippage_bps:    float = 5.0,
        spread_cost_bps: float = 2.0,
        latency_ms:      float = 100.0,
        position_limit:  float = 1.0,
        allow_short:     bool  = False,
        min_bars_between_trades: int = 1,
        max_dd_limit:    float = 0.15,
        exec_surrogate:  ExecutionSurrogate | None = None,
        seed:            int   = 42,
    ):
        self.initial_capital = initial_capital
        self.commission_pct  = commission_pct
        self.slippage_bps     = slippage_bps
        self.spread_cost_bps  = spread_cost_bps
        self.latency_ms       = latency_ms
        self.position_limit   = position_limit
        self.allow_short      = allow_short
        self.min_bars         = min_bars_between_trades
        self.max_dd_limit     = max_dd_limit
        self.exec_surrogate   = exec_surrogate or ExecutionSurrogate(seed=seed)
        self.rng              = np.random.default_rng(seed)

    def run(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        strategy_params: dict | None = None,
        stop_loss_pct:  float = 0.02,
        take_profit_pct: float = 0.05,
        trailing_stop_pct: float = 0.0,
        time_exit_bars:  int = 0,
        stochastic_exec: bool = True,
    ) -> BacktestResult:
        """
        Run a full backtest on a DataFrame that already has engineered features.

        Args:
            df:              OHLCV DataFrame with features (from FeatureEngine)
            strategy_name:   one of "momentum" | "mean_reversion"
            strategy_params: strategy-specific parameters (from YAML config)
            stop_loss_pct:   fraction (0.02 = 2%)
            take_profit_pct: fraction (0.05 = 5%)
            trailing_stop_pct: trailing stop fraction
            time_exit_bars:  exit after N bars if > 0
            stochastic_exec: add random fill/no-fill variation

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        params = strategy_params or {}
        n = len(df)
        if n < 20:
            raise ValueError(f"DataFrame too short: {n} bars (minimum 20)")

        # ── Generate signals ──────────────────────────────────────────────
        signal_fn = STRATEGY_SIGNALS.get(strategy_name)
        if not signal_fn:
            raise ValueError(f"Unknown strategy: {strategy_name}. "
                             f"Available: {list(STRATEGY_SIGNALS.keys())}")

        signals = signal_fn(df, params)

        # ── Initialise state ───────────────────────────────────────────────
        state = BacktestState(
            capital=self.initial_capital,
            position=0.0,
            position_avg=0.0,
            entry_bar=-1,
            pending_order=None,
        )

        equity_curve  = []
        timestamps   = []
        trade_log    = []   # detailed per-bar log
        completed_trades: list[Trade] = []

        # ADV for execution model (approximate from first 20 bars)
        adv = float(df["volume"].iloc[:20].mean()) if "volume" in df.columns else 1e6

        # Per-bar loop (strictly past-only)
        bars_since_trade = 0
        trailing_stop_price = 0.0

        for i in range(n):
            row     = df.iloc[i]
            price   = float(row["close"])
            ts      = str(row.get("timestamp", i))
            vol_i   = float(row.get("volatility", 0.01))
            sig     = int(signals.iloc[i])
            rng_val = float(df.get(f"return_{1}d", pd.Series(0, index=df.index)).iloc[i]) if "return_1d" in df.columns else 0.0

            bars_since_trade += 1

            # ── 1. Entry logic ─────────────────────────────────────────────
            prev_close = state.prev_close
            if state.position == 0 and sig != 0:
                if bars_since_trade < self.min_bars:
                    pass
                elif sig == 1 and state.capital > 0:
                    self._open_position(
                        state, row, price, vol_i, adv,
                        OrderSide.BUY, sig, stochastic_exec,
                        trade_log, i, ts, params,
                    )
                    trailing_stop_price = 0.0
                elif sig == -1 and state.capital > 0 and self.allow_short:
                    self._open_position(
                        state, row, price, vol_i, adv,
                        OrderSide.SELL, sig, stochastic_exec,
                        trade_log, i, ts, params,
                    )
                    trailing_stop_price = 0.0

            # ── 2. Exit logic ──────────────────────────────────────────────
            elif state.position != 0:
                should_exit = False
                exit_reason = ""

                # Time exit
                if time_exit_bars > 0 and (i - state.entry_bar) >= time_exit_bars:
                    should_exit = True
                    exit_reason = "time_exit"

                # Stop-loss
                if stop_loss_pct > 0:
                    if state.position > 0:
                        if price <= state.position_avg * (1 - stop_loss_pct):
                            should_exit = True
                            exit_reason = "stop_loss"
                    else:
                        if price >= state.position_avg * (1 + stop_loss_pct):
                            should_exit = True
                            exit_reason = "stop_loss"

                # Take-profit (independent of stop-loss)
                if not should_exit and take_profit_pct > 0:
                    if state.position > 0:
                        if price >= state.position_avg * (1 + take_profit_pct):
                            should_exit = True
                            exit_reason = "take_profit"
                    else:
                        if price <= state.position_avg * (1 - take_profit_pct):
                            should_exit = True
                            exit_reason = "take_profit"

                # Trailing stop
                if trailing_stop_pct > 0 and not should_exit:
                    if state.position > 0:
                        if price > trailing_stop_price or trailing_stop_price == 0:
                            trailing_stop_price = price * (1 - trailing_stop_pct)
                        elif price <= trailing_stop_price:
                            should_exit = True
                            exit_reason = "trailing_stop"

                # Signal reversal exit
                if not should_exit and sig != 0 and sig != int(np.sign(state.position)):
                    should_exit = True
                    exit_reason = "signal_flip"

                if should_exit:
                    self._close_position(
                        state, row, price, vol_i, adv,
                        stochastic_exec, completed_trades,
                        trade_log, i, ts, exit_reason,
                    )
                    bars_since_trade = 0

            # ── 3. Update equity ─────────────────────────────────────────────
            # Equity = cash + unrealized PnL (position value vs cost basis)
            # If long: equity = capital + position * (current_price - avg_price)
            # If short: equity = capital + |position| * (avg_price - current_price)
            direction = 1 if state.position > 0 else -1 if state.position < 0 else 0
            unrealised_pnl = direction * (price - state.position_avg) * abs(state.position)
            equity_bar = state.capital + unrealised_pnl

            # Cap drawdown to max DD limit
            if self.initial_capital > 0:
                dd = (self.initial_capital - equity_bar) / self.initial_capital
                if dd > self.max_dd_limit:
                    equity_bar = self.initial_capital * (1 - self.max_dd_limit)
                    if state.position != 0:
                        self._close_position(
                            state, row, price, vol_i, adv,
                            stochastic_exec, completed_trades,
                            trade_log, i, ts, "max_dd_stop",
                        )

            equity_curve.append(equity_bar)
            timestamps.append(ts)

            # Store price for next bar's equity continuity
            state.prev_close = price

        # ── Close any open position at end of backtest ─────────────────────
        if state.position != 0:
            row   = df.iloc[-1]
            price = float(row["close"])
            self._close_position(
                state, row, price, float(row.get("volatility", 0.01)), adv,
                stochastic_exec, completed_trades,
                trade_log, len(df) - 1, str(df.iloc[-1].get("timestamp", "")),
                "end_of_backtest",
            )
            equity_curve[-1] = state.capital

        # ── Compute drawdown curve ─────────────────────────────────────────
        eq = pd.Series(equity_curve)
        running_max = eq.cummax()
        drawdown_curve = ((eq - running_max) / running_max * 100).tolist()

        # ── Compute metrics ────────────────────────────────────────────────
        trades_df = pd.DataFrame([asdict(t) for t in completed_trades]) if completed_trades else None
        metrics = compute_metrics(equity_curve, trades_df)
        metrics_dict = metrics_to_dict(metrics)

        # ── Assemble result ───────────────────────────────────────────────
        trades_as_dict = [asdict(t) for t in completed_trades]

        return BacktestResult(
            strategy_name=strategy_name,
            equity_curve=[round(v, 2) for v in equity_curve],
            timestamps=timestamps,
            trades=trades_as_dict,
            metrics=metrics_dict,
            drawdown_curve=[round(v, 4) for v in drawdown_curve],
            config={
                "strategy_name":    strategy_name,
                "strategy_params":  params,
                "stop_loss_pct":    stop_loss_pct,
                "take_profit_pct":  take_profit_pct,
                "trailing_stop_pct": trailing_stop_pct,
                "time_exit_bars":  time_exit_bars,
                "initial_capital":  self.initial_capital,
                "commission_pct":   self.commission_pct,
                "slippage_bps":     self.slippage_bps,
            },
            split_period=f"{timestamps[0]} -- {timestamps[-1]}",
            n_bars=n,
            trade_log=trade_log,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _open_position(
        self,
        state: BacktestState,
        row: pd.Series,
        price: float,
        vol_i: float,
        adv: float,
        side: OrderSide,
        sig: int,
        stochastic: bool,
        log: list,
        bar_idx: int,
        ts: str,
        params: dict,
    ) -> None:
        max_pos_pct = params.get("max_position_pct", self.position_limit)
        capital     = state.capital
        if state.position != 0:
            unrealised = state.position * (price - state.position_avg)
            capital    = capital + unrealised
        order_value = capital * max_pos_pct
        size        = order_value / price

        result = self.exec_surrogate.simulate_fill(
            signal_price=price,
            order_size=size,
            direction=+1 if sig == 1 else -1,
            current_vol=vol_i,
            adv=adv,
            spread_bps=self.spread_cost_bps,
            latency_ms=self.latency_ms,
            stochastic=stochastic,
        )

        if result.filled:
            cost = self.exec_surrogate.compute_transaction_cost(
                order_value, self.commission_pct, self.slippage_bps
            )
            state.capital -= cost
            state.position = size * (1 if sig == 1 else -1)
            state.position_avg = result.fill_price
            state.entry_bar   = bar_idx

            log.append({
                "bar": bar_idx,
                "timestamp": ts,
                "action": "OPEN",
                "side": side.value,
                "signal_price": round(price, 2),
                "fill_price": round(result.fill_price, 2),
                "size": round(size, 4),
                "slippage_bps": round(result.slippage_bps, 2),
                "fill_prob": round(result.fill_prob, 3),
            })

    def _close_position(
        self,
        state: BacktestState,
        row: pd.Series,
        price: float,
        vol_i: float,
        adv: float,
        stochastic: bool,
        trades: list[Trade],
        log: list,
        bar_idx: int,
        ts: str,
        reason: str,
    ) -> None:
        if state.position == 0:
            return

        size      = abs(state.position)
        direction = +1 if state.position > 0 else -1

        result = self.exec_surrogate.simulate_fill(
            signal_price=price,
            order_size=size,
            direction=direction,
            current_vol=vol_i,
            adv=adv,
            spread_bps=self.spread_cost_bps,
            latency_ms=self.latency_ms,
            stochastic=stochastic,
        )

        if result.filled:
            cost = self.exec_surrogate.compute_transaction_cost(
                size * result.fill_price, self.commission_pct, self.slippage_bps
            )
            pnl = direction * (result.fill_price - state.position_avg) * size - cost

            # Retrieve entry timestamp from log
            open_entries = [e for e in log if e["bar"] == state.entry_bar and e["action"] == "OPEN"]
            entry_ts = open_entries[-1]["timestamp"] if open_entries else str(bar_idx)

            trade = Trade(
                entry_date  = entry_ts,
                exit_date   = ts,
                side        = "buy" if direction > 0 else "sell",
                entry_price = round(state.position_avg, 4),
                exit_price  = round(result.fill_price, 4),
                size        = round(size, 4),
                pnl         = round(pnl, 2),
                return_pct  = round(direction * (result.fill_price / state.position_avg - 1) * 100, 3),
                hold_bars   = bar_idx - state.entry_bar,
                notional    = round(size * result.fill_price, 2),
                commission  = round(cost, 2),
                slippage    = round(result.slippage_bps * size, 2),
                fill_prob   = round(result.fill_prob, 3),
            )
            trades.append(trade)

            state.capital += pnl
            log.append({
                "bar": bar_idx,
                "timestamp": ts,
                "action": "CLOSE",
                "reason": reason,
                "exit_price": round(result.fill_price, 2),
                "pnl": round(pnl, 2),
                "fill_prob": round(result.fill_prob, 3),
            })

        state.position  = 0.0
        state.position_avg = 0.0
        state.entry_bar  = -1
