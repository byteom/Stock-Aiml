"""Pydantic schemas for FastAPI request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ─── Strategy config ─────────────────────────────────────────────────────────────

class StrategyParams(BaseModel):
    """Strategy-specific parameters."""
    # Momentum
    rsi_buy_threshold:  float = 30.0
    rsi_sell_threshold: float = 70.0
    lookback_short:     int   = 5
    lookback_long:      int   = 20
    min_return_diff:    float = 0.02
    # Position sizing
    max_position_pct:   float = 1.0
    vol_lookback:        int   = 20
    # Risk
    stop_loss_pct:      float = 0.02
    take_profit_pct:     float = 0.05
    trailing_stop_pct:  float = 0.0
    time_exit_bars:      int   = 0


# ─── Request models ─────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    """POST /api/v1/backtest"""
    data_path:       str | None = None
    strategy_name:  str = "momentum"
    strategy_params: StrategyParams = Field(default_factory=StrategyParams)
    initial_capital: float = 1_000_000.0
    commission_pct:  float = 0.001
    slippage_bps:    float = 5.0
    split_id:        int   = 0
    stochastic_exec: bool  = True


class StressTestRequest(BaseModel):
    """POST /api/v1/stress-test"""
    backtest_run_id: str | None = None
    data_path:       str | None = None
    strategy_name:   str = "momentum"
    strategy_params: StrategyParams = Field(default_factory=StrategyParams)
    n_scenarios:      int = 50
    scenario_types:  list[str] | None = None


class OptimizeRequest(BaseModel):
    """POST /api/v1/optimize"""
    data_path:       str | None = None
    strategy_name:   str = "momentum"
    param_grid:      dict[str, list] | None = None
    mode:            str = "grid_search"   # "grid_search" | "ppo"


class ExplainRequest(BaseModel):
    """POST /api/v1/explain"""
    backtest_run_id: str | None = None
    backtest_result_json: dict | None = None
    target_metric:   str = "max_drawdown"
    n_counterfactuals: int = 10


# ─── Response models ───────────────────────────────────────────────────────────

class MetricsSummary(BaseModel):
    annualized_return: float
    sharpe_ratio:      float
    sortino_ratio:     float
    max_drawdown:      float
    max_drawdown_pct: float
    cvar_95:           float
    win_rate:          float
    total_trades:      int
    profit_factor:     float
    volatility:        float
    calmar_ratio:      float


class TradeEntry(BaseModel):
    entry_date:   str
    exit_date:    str
    side:         str
    entry_price:  float
    exit_price:   float
    size:         float
    pnl:          float
    return_pct:   float
    hold_bars:    int
    notional:     float
    commission:   float
    slippage:     float


class BacktestResponse(BaseModel):
    """Response from a backtest run."""
    run_id:          str
    strategy_name:   str
    split_period:    str
    n_bars:          int
    equity_curve:    list[float]
    drawdown_curve:  list[float]
    timestamps:      list[str]
    metrics:         MetricsSummary
    trades:          list[TradeEntry]
    config:           dict


class StressTestResponse(BaseModel):
    """Response from a stress test run."""
    run_id:          str
    strategy_name:   str
    n_scenarios:     int
    results:         list[dict]   # per-scenario metrics
    worst_case:      MetricsSummary
    best_case:       MetricsSummary
    avg_case:        MetricsSummary
    scenario_descriptions: list[str]


class OptimizeResponse(BaseModel):
    """Response from RL/grid optimization."""
    run_id:          str
    strategy_name:   str
    mode:            str
    best_params:     dict
    best_metrics:    MetricsSummary
    n_trials:        int
    all_trials:      list[dict] | None = None


class ExplainResponse(BaseModel):
    """Response from counterfactual analysis."""
    run_id:           str
    target_metric:    str
    baseline_metrics: dict
    interventions:    list[dict]
    feature_importance: dict
    summary:          str


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
