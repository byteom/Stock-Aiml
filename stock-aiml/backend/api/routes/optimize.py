"""POST /api/v1/optimize — RL/grid-search optimization."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.schemas import OptimizeRequest, OptimizeResponse, MetricsSummary
from backend.services.data_loader import DataLoader
from backend.services.feature_engineering import FeatureEngine
from backend.services.rl_optimizer import RLOptimizer

router = APIRouter(prefix="/api/v1", tags=["optimize"])


@router.post("/optimize", response_model=OptimizeResponse)
async def run_optimize(req: OptimizeRequest) -> OptimizeResponse:
    """
    Optimize strategy parameters using grid-search (MVP) or PPO RL.

    Grid-search mode: exhaustive search over param_grid
    PPO mode: uses trained PPO agent to find optimal parameters
    """
    if req.data_path:
        path = Path(req.data_path)
    else:
        path = Path(__file__).parents[2] / "data" / "raw" / "data.csv"
        if not path.exists():
            path = Path(__file__).parents[4] / "data.csv"

    try:
        loader = DataLoader(path)
        loader.load_csv(path)
    except Exception:
        raise HTTPException(status_code=500, detail="Data file not found on server. Please use the Streamlit UI to upload data.")
    splits = loader.create_walk_forward_splits(n_splits=3)
    test_df = splits[0]["test"]

    engine = FeatureEngine()
    test_df = engine.transform(test_df)

    # Always initialize optimizer — it handles both grid_search and ppo modes
    optimizer = RLOptimizer()

    if req.mode == "grid_search" or req.param_grid:
        param_grid = req.param_grid or {
            "lookback_short": [3, 5, 7],
            "lookback_long":  [15, 20, 25],
            "stop_loss_pct":  [0.01, 0.02, 0.03],
            "take_profit_pct": [0.03, 0.05, 0.07],
            "max_position_pct": [0.5, 1.0],
        }

        result = optimizer.grid_search(
            df=test_df,
            strategy_name=req.strategy_name,
            param_grid=param_grid,
        )

        return OptimizeResponse(
            run_id=str(uuid.uuid4())[:8],
            strategy_name=req.strategy_name,
            mode="grid_search",
            best_params=result.best_params,
            best_metrics=MetricsSummary(
                annualized_return=result.best_metrics.get("annualized_return", 0.0),
                sharpe_ratio=result.best_metrics.get("sharpe_ratio", 0.0),
                sortino_ratio=result.best_metrics.get("sortino_ratio", 0.0),
                max_drawdown=result.best_metrics.get("max_drawdown", 0.0),
                max_drawdown_pct=result.best_metrics.get("max_drawdown_pct", 0.0),
                cvar_95=result.best_metrics.get("cvar_95", 0.0),
                win_rate=result.best_metrics.get("win_rate", 0.0),
                total_trades=result.best_metrics.get("total_trades", 0),
                profit_factor=result.best_metrics.get("profit_factor", 0.0),
                volatility=result.best_metrics.get("volatility", 0.0),
                calmar_ratio=result.best_metrics.get("calmar_ratio", 0.0),
            ),
            n_trials=len(result.all_param_trials),
            all_trials=result.all_param_trials[:50],
        )

    elif req.mode == "ppo":
        # Use trained PPO agent
        result = optimizer.optimize(df=test_df)

        return OptimizeResponse(
            run_id=str(uuid.uuid4())[:8],
            strategy_name=req.strategy_name,
            mode="ppo",
            best_params=result["recommended_params"],
            best_metrics=MetricsSummary(
                annualized_return=result["metrics"].get("annualized_return", 0.0),
                sharpe_ratio=result["metrics"].get("sharpe_ratio", 0.0),
                sortino_ratio=result["metrics"].get("sortino_ratio", 0.0),
                max_drawdown=result["metrics"].get("max_drawdown", 0.0),
                max_drawdown_pct=result["metrics"].get("max_drawdown_pct", 0.0),
                cvar_95=result["metrics"].get("cvar_95", 0.0),
                win_rate=result["metrics"].get("win_rate", 0.0),
                total_trades=result["metrics"].get("total_trades", 0),
                profit_factor=result["metrics"].get("profit_factor", 0.0),
                volatility=result["metrics"].get("volatility", 0.0),
                calmar_ratio=result["metrics"].get("calmar_ratio", 0.0),
            ),
            n_trials=result.get("num_trades", 0),
            all_trials=None,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")