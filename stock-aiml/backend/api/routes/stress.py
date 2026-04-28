"""POST /api/v1/stress-test — adversarial stress testing."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.schemas import StressTestRequest, StressTestResponse, MetricsSummary
from backend.services.data_loader import DataLoader
from backend.services.feature_engineering import FeatureEngine
from backend.services.backtester import Backtester
from backend.services.execution_model import ExecutionSurrogate
from backend.services.stress_generator import AdversarialMarketGenerator, ScenarioType
from backend.services.metrics import compute_metrics, metrics_to_dict

router = APIRouter(prefix="/api/v1", tags=["stress-test"])


@router.post("/stress-test", response_model=StressTestResponse)
async def run_stress_test(req: StressTestRequest) -> StressTestResponse:
    """
    Run adversarial stress testing on a strategy.

    1. Load data and select split
    2. Engineer features on the test window
    3. Generate N adversarial scenarios using AMG
    4. For each scenario: run backtester and collect metrics
    5. Return worst/best/avg case analysis
    """
    # Load data
    if req.data_path:
        path = Path(req.data_path)
    else:
        path = Path(__file__).parents[2] / "data" / "raw" / "data.csv"
        if not path.exists():
            path = Path(__file__).parents[4] / "data.csv"

    loader = DataLoader(path)
    loader.load_csv(path)
    splits = loader.create_walk_forward_splits(n_splits=3)
    if req.split_id >= len(splits):
        raise HTTPException(status_code=400, detail=f"split_id out of range")

    test_df = splits[req.split_id]["test"]
    engine  = FeatureEngine()
    test_df = engine.transform(test_df)

    exec_surrogate = ExecutionSurrogate()
    bt = Backtester(
        initial_capital=1_000_000.0,
        commission_pct=0.001,
        slippage_bps=5.0,
        exec_surrogate=exec_surrogate,
    )

    strat_params = req.strategy_params.model_dump()

    # Generate adversarial scenarios
    amg = AdversarialMarketGenerator()
    scenario_types = []
    for t in (req.scenario_types or []):
        try:
            scenario_types.append(ScenarioType(t))
        except (ValueError, TypeError):
            pass

    scenarios = amg.generate_n_scenarios(
        test_df,
        n=req.n_scenarios,
        scenario_types=scenario_types or None,
    )

    scenario_results = []
    for sc in scenarios:
        df_scen = sc.generated_df
        if df_scen is None or len(df_scen) < 20:
            continue

        # Re-run feature engineering on modified data
        df_fe = engine.transform(df_scen)

        try:
            result = bt.run(
                df=df_fe,
                strategy_name=req.strategy_name,
                strategy_params=strat_params,
                stochastic_exec=False,
            )
        except Exception:
            continue

        m = result.metrics
        scenario_results.append({
            "scenario_name":       sc.name,
            "scenario_type":       sc.scenario_type.value,
            "magnitude":           sc.magnitude,
            "plausibility_score":  sc.plausibility_score,
            "metrics":             m,
            "total_return":        m["total_return"],
            "sharpe_ratio":        m["sharpe_ratio"],
            "max_drawdown":        m["max_drawdown"],
        })

    if not scenario_results:
        raise HTTPException(status_code=500, detail="All stress scenarios failed")

    # Aggregate
    import numpy as np
    worst_case = min(scenario_results, key=lambda x: x["metrics"]["annualized_return"])
    best_case  = max(scenario_results, key=lambda x: x["metrics"]["annualized_return"])

    avg_metrics = {
        "annualized_return": np.mean([r["metrics"]["annualized_return"] for r in scenario_results]),
        "sharpe_ratio":      np.mean([r["metrics"]["sharpe_ratio"] for r in scenario_results]),
        "sortino_ratio":     np.mean([r["metrics"]["sortino_ratio"] for r in scenario_results]),
        "max_drawdown":      np.mean([r["metrics"]["max_drawdown"] for r in scenario_results]),
        "max_drawdown_pct":  np.mean([r["metrics"]["max_drawdown_pct"] for r in scenario_results]),
        "cvar_95":           np.mean([r["metrics"]["cvar_95"] for r in scenario_results]),
        "win_rate":          np.mean([r["metrics"]["win_rate"] for r in scenario_results]),
        "total_trades":      int(np.mean([r["metrics"]["total_trades"] for r in scenario_results])),
        "profit_factor":     np.mean([r["metrics"]["profit_factor"] for r in scenario_results]),
        "volatility":        np.mean([r["metrics"]["volatility"] for r in scenario_results]),
        "calmar_ratio":      np.mean([r["metrics"]["calmar_ratio"] for r in scenario_results]),
    }

    descriptions = [
        amg.get_scenario_description(sc) if sc in scenarios else ""
        for sc in scenarios[:5]
    ]

    return StressTestResponse(
        run_id=str(uuid.uuid4())[:8],
        strategy_name=req.strategy_name,
        n_scenarios=len(scenario_results),
        results=scenario_results,
        worst_case=MetricsSummary(**worst_case["metrics"]),
        best_case=MetricsSummary(**best_case["metrics"]),
        avg_case=MetricsSummary(**avg_metrics),
        scenario_descriptions=descriptions,
    )
