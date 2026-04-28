"""POST /api/v1/explain — counterfactual analysis."""
from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException

from backend.api.schemas import ExplainRequest, ExplainResponse
from backend.services.counterfactual import CounterfactualEngine, SurrogateModel

router = APIRouter(prefix="/api/v1", tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
async def run_explain(req: ExplainRequest) -> ExplainResponse:
    """
    Generate counterfactual analysis for a backtest result.

    Takes either a backtest_run_id or the full backtest_result_json.
    Returns ranked actionable interventions with expected metric improvements.
    """
    # Reconstruct a minimal backtest result from JSON
    if req.backtest_result_json:
        class FakeResult:
            def __init__(self, data):
                self.strategy_name = data.get("strategy_name", "unknown")
                self.metrics = data.get("metrics", {})

        result = FakeResult(req.backtest_result_json)
    elif req.backtest_run_id:
        # In a real system, fetch from DB by run_id
        raise HTTPException(status_code=404, detail=f"Backtest run {req.backtest_run_id} not found in DB")
    else:
        raise HTTPException(status_code=400, detail="Either backtest_run_id or backtest_result_json required")

    engine = CounterfactualEngine(
        surrogate_model=SurrogateModel(),
        n_counterfactuals=req.n_counterfactuals,
    )

    cf_result = engine.analyze(result, target_metric=req.target_metric)
    return ExplainResponse(
        run_id=str(uuid.uuid4())[:8],
        target_metric=cf_result.target_metric,
        baseline_metrics=cf_result.baseline_metrics,
        interventions=[
            {
                "variable": iv.variable,
                "original_value": iv.original_value,
                "counterfactual_value": iv.counterfactual_value,
                "delta": iv.delta,
                "expected_metric": iv.expected_metric,
                "improvement": round(iv.improvement * 100, 3),
                "actionable": iv.actionable,
                "rank": iv.rank,
            }
            for iv in cf_result.interventions
        ],
        feature_importance={
            k: round(v, 4) for k, v in cf_result.feature_importance.items()
        },
        summary=cf_result.summary,
    )
