"""Performance metrics computation — no lookahead, using only equity series."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class PerformanceMetrics(NamedTuple):
    """All key performance indicators for a backtest run."""
    total_return:       float
    annualized_return:  float
    sharpe_ratio:       float
    sortino_ratio:      float
    max_drawdown:       float       # fraction (positive), e.g. 0.18 = 18%
    max_drawdown_pct:   float       # same as max_drawdown, but as %
    max_drawdown_duration: int    # bars in longest drawdown
    cvar_95:            float       # CVaR at 95% confidence (negative = loss)
    cvar_99:            float
    win_rate:           float
    loss_rate:          float
    total_trades:       int
    avg_trade_pnl:      float
    profit_factor:      float
    turnover:           float       # avg daily turnover fraction
    volatility:         float       # daily vol of returns
    calmar_ratio:       float       # annualized_return / max_drawdown


def compute_metrics(
    equity_curve: pd.Series | list[float],
    trades: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> PerformanceMetrics:
    """
    Compute a full suite of performance metrics from an equity curve.

    Args:
        equity_curve: daily portfolio value series
        trades: optional trade log DataFrame (for per-trade stats)
        risk_free_rate: annual risk-free rate (default 0)
        periods_per_year: bars per year (default 252 for daily data)

    All calculations are rolling and use closed-form formulas —
    no future data is referenced.
    """
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)

    equity = equity_curve.dropna()
    if len(equity) < 2:
        raise ValueError("equity_curve must have at least 2 data points")

    # ── Daily returns ──────────────────────────────────────────────────────
    returns = equity.pct_change().fillna(0)
    log_returns = np.log(equity / equity.shift(1)).fillna(0)

    # ── Return metrics ─────────────────────────────────────────────────────
    total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1)

    # Annualized return: simple CAGR when initial capital is reasonable
    if equity.iloc[0] < 100.0:
        ann_return = 0.0
    elif len(equity) < 2:
        ann_return = 0.0
    else:
        n_years = len(equity) / periods_per_year
        ann_return = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1)

    # ── Volatility ─────────────────────────────────────────────────────────
    daily_vol = returns.std()
    ann_vol   = daily_vol * np.sqrt(periods_per_year)

    # ── Sharpe ratio ───────────────────────────────────────────────────────
    excess_return = returns - risk_free_rate / periods_per_year
    if ann_vol > 0:
        sharpe = float(excess_return.mean() / daily_vol) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # ── Sortino ratio ──────────────────────────────────────────────────────
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
    else:
        downside_std = daily_vol  # fallback if no losses
    if downside_std > 0:
        sortino = float(excess_return.mean() / downside_std) * np.sqrt(periods_per_year)
    else:
        sortino = 0.0

    # ── Drawdown series ────────────────────────────────────────────────────
    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    max_dd      = float(drawdown.min())   # negative number
    max_dd_pct  = abs(max_dd * 100)

    # ── Drawdown duration ─────────────────────────────────────────────────
    in_dd = drawdown < 0
    dd_durations = []
    current_dd_len = 0
    for v in in_dd:
        if v:
            current_dd_len += 1
        else:
            if current_dd_len > 0:
                dd_durations.append(current_dd_len)
            current_dd_len = 0
    if current_dd_len > 0:
        dd_durations.append(current_dd_len)
    max_dd_duration = int(max(dd_durations)) if dd_durations else 0

    # ── CVaR (Conditional Value at Risk) ─────────────────────────────────
    def cvar(ret: pd.Series, confidence: float = 0.95) -> float:
        var_threshold = ret.quantile(1 - confidence)
        return float(ret[ret <= var_threshold].mean())

    cvar_95 = cvar(returns, 0.95)
    cvar_99 = cvar(returns, 0.99)

    # ── Trade-level metrics ────────────────────────────────────────────────
    if trades is not None and len(trades) > 0 and "pnl" in trades.columns:
        pnls  = trades["pnl"].dropna()
        wins  = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        total_trades = len(pnls)
        avg_pnl     = float(pnls.mean()) if len(pnls) > 0 else 0.0
        win_rate    = float(len(wins) / len(pnls)) if len(pnls) > 0 else 0.0
        loss_rate   = float(len(losses) / len(pnls)) if len(pnls) > 0 else 0.0
        profit_factor = (
            float(abs(wins.sum()) / abs(losses.sum()))
            if len(losses) > 0 and losses.sum() != 0 else 0.0
        )
    else:
        total_trades = 0
        avg_pnl      = 0.0
        win_rate     = 0.0
        loss_rate    = 0.0
        profit_factor = 0.0

    # ── Turnover ──────────────────────────────────────────────────────────
    if trades is not None and len(trades) > 0 and "notional" in trades.columns:
        daily_notional = trades.groupby("exit_date")["notional"].sum()
        avg_capital    = equity.mean()
        turnover = float((daily_notional.sum() / len(equity)) / avg_capital) if avg_capital > 0 else 0.0
    else:
        turnover = 0.0

    # ── Calmar ratio ──────────────────────────────────────────────────────
    calmar = float(ann_return / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

    return PerformanceMetrics(
        total_return=float(total_return),
        annualized_return=float(ann_return),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(abs(max_dd)),
        max_drawdown_pct=float(max_dd_pct),
        max_drawdown_duration=max_dd_duration,
        cvar_95=float(cvar_95),
        cvar_99=float(cvar_99),
        win_rate=float(win_rate),
        loss_rate=float(loss_rate),
        total_trades=total_trades,
        avg_trade_pnl=float(avg_pnl),
        profit_factor=float(profit_factor),
        turnover=float(turnover),
        volatility=float(ann_vol),
        calmar_ratio=float(calmar),
    )


def metrics_to_dict(m: PerformanceMetrics) -> dict:
    """Convert NamedTuple to dict for JSON serialization."""
    return {
        "total_return":          round(m.total_return * 100, 3),
        "annualized_return":     round(m.annualized_return * 100, 3),
        "sharpe_ratio":          round(m.sharpe_ratio, 3),
        "sortino_ratio":         round(m.sortino_ratio, 3),
        "max_drawdown":          round(m.max_drawdown * 100, 3),
        "max_drawdown_pct":      round(m.max_drawdown_pct, 3),
        "max_drawdown_duration": m.max_drawdown_duration,
        "cvar_95":               round(m.cvar_95 * 100, 3),
        "cvar_99":               round(m.cvar_99 * 100, 3),
        "win_rate":              round(m.win_rate * 100, 2),
        "loss_rate":             round(m.loss_rate * 100, 2),
        "total_trades":          m.total_trades,
        "avg_trade_pnl":         round(m.avg_trade_pnl, 2),
        "profit_factor":         round(m.profit_factor, 3),
        "turnover":              round(m.turnover, 4),
        "volatility":            round(m.volatility * 100, 3),
        "calmar_ratio":          round(m.calmar_ratio, 3),
    }
