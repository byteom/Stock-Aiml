"""POST /api/v1/backtest — run a backtest."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    MetricsSummary,
    TradeEntry,
)
from backend.services.data_loader import DataLoader
from backend.services.feature_engineering import FeatureEngine
from backend.services.backtester import Backtester
from backend.services.execution_model import ExecutionSurrogate

router = APIRouter(prefix="/api/v1", tags=["backtest"])


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """
    Run an execution-aware backtest on uploaded data.

    Steps:
      1. Load CSV from data_path (or use last uploaded)
      2. Create walk-forward splits and select the requested split_id
      3. Engineer features on the selected window
      4. Run the backtester with the given strategy params
      5. Return equity curve, trades, and metrics
    """
    # ── 1. Load data ─────────────────────────────────────────────────────────
    if req.data_path:
        path = Path(req.data_path)
    else:
        # Default: look for data.csv in project root
        path = Path(__file__).parents[2] / "data" / "raw" / "data.csv"
        if not path.exists():
            # Fallback: sibling path
            path = Path(__file__).parents[4] / "data.csv"

    loader = DataLoader(path)
    try:
        loader.load_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Data file not found on server. Please use the Streamlit UI to upload data.")

    # ── 2. Walk-forward split ───────────────────────────────────────────────
    splits = loader.create_walk_forward_splits(n_splits=3)
    if req.split_id >= len(splits):
        raise HTTPException(
            status_code=400,
            detail=f"split_id {req.split_id} out of range (have {len(splits)} splits)"
        )

    split = splits[req.split_id]
    test_df = split["test"]

    # ── 3. Feature engineering ───────────────────────────────────────────────
    engine = FeatureEngine()
    test_df = engine.transform(test_df)

    # ── 4. Run backtester ────────────────────────────────────────────────────
    exec_surrogate = ExecutionSurrogate(
        impact_coeff=0.5,
        fill_rate=0.85,
        vol_impact_mult=1.2,
        spread_factor=0.5,
    )

    bt = Backtester(
        initial_capital=req.initial_capital,
        commission_pct=req.commission_pct,
        slippage_bps=req.slippage_bps,
        spread_cost_bps=2.0,
        latency_ms=100.0,
        exec_surrogate=exec_surrogate,
    )

    strat_params = req.strategy_params.model_dump()

    try:
        result = bt.run(
            df=test_df,
            strategy_name=req.strategy_name,
            strategy_params=strat_params,
            stop_loss_pct=strat_params.get("stop_loss_pct", 0.02),
            take_profit_pct=strat_params.get("take_profit_pct", 0.05),
            trailing_stop_pct=strat_params.get("trailing_stop_pct", 0.0),
            time_exit_bars=strat_params.get("time_exit_bars", 0),
            stochastic_exec=req.stochastic_exec,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    # ── 5. Assemble response ─────────────────────────────────────────────────
    run_id = str(uuid.uuid4())[:8]

    metrics_dict = result.metrics
    return BacktestResponse(
        run_id=run_id,
        strategy_name=result.strategy_name,
        split_period=result.split_period,
        n_bars=result.n_bars,
        equity_curve=result.equity_curve,
        drawdown_curve=result.drawdown_curve,
        timestamps=result.timestamps,
        metrics=MetricsSummary(
            annualized_return=metrics_dict["annualized_return"],
            sharpe_ratio=metrics_dict["sharpe_ratio"],
            sortino_ratio=metrics_dict["sortino_ratio"],
            max_drawdown=metrics_dict["max_drawdown"],
            max_drawdown_pct=metrics_dict["max_drawdown_pct"],
            cvar_95=metrics_dict["cvar_95"],
            win_rate=metrics_dict["win_rate"],
            total_trades=metrics_dict["total_trades"],
            profit_factor=metrics_dict["profit_factor"],
            volatility=metrics_dict["volatility"],
            calmar_ratio=metrics_dict["calmar_ratio"],
        ),
        trades=[TradeEntry(**t) for t in result.trades],
        config=result.config,
    )
