"""CLI script to run a backtest from the command line."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from backend.services.data_loader import DataLoader
from backend.services.feature_engineering import FeatureEngine
from backend.services.backtester import Backtester
from backend.services.execution_model import ExecutionSurrogate
from backend.services.metrics import compute_metrics, metrics_to_dict


def main():
    # ── Locate data ──────────────────────────────────────────────────────────
    data_path = Path("C:/Users/anwee/Desktop_1/Learning-Season/Stock-Aiml/data.csv")

    print(f"[Data] Loading from: {data_path}")

    # ── Load & split ────────────────────────────────────────────────────────
    loader = DataLoader(data_path)
    loader.load_csv(data_path)
    splits = loader.create_walk_forward_splits(n_splits=3)
    test_split = splits[0]

    print(f"  Train period: {test_split['train_period']}")
    print(f"  Val period:   {test_split['val_period']}")
    print(f"  Test period:  {test_split['test_period']}")

    test_df = test_split["test"]

    # ── Feature engineering ─────────────────────────────────────────────────
    engine = FeatureEngine()
    test_df = engine.transform(test_df)
    print(f"\n[Done] Engineered {len(engine.get_feature_columns())} features")

    # ── Run backtest ────────────────────────────────────────────────────────
    exec_surrogate = ExecutionSurrogate(seed=42)
    bt = Backtester(
        initial_capital=1_000_000.0,
        commission_pct=0.001,
        slippage_bps=5.0,
        exec_surrogate=exec_surrogate,
    )

    result = bt.run(
        df=test_df,
        strategy_name="momentum",
        strategy_params={
            "rsi_buy_threshold":  30.0,
            "rsi_sell_threshold": 70.0,
            "lookback_short":     5,
            "lookback_long":      20,
            "min_return_diff":    0.02,
            "max_position_pct":   1.0,
            "stop_loss_pct":      0.02,
            "take_profit_pct":    0.05,
            "trailing_stop_pct":  0.0,
            "time_exit_bars":     0,
        },
        stochastic_exec=True,
    )

    # ── Print results ────────────────────────────────────────────────────────
    m = result.metrics
    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS — {result.strategy_name}")
    print("=" * 60)
    print(f"  Period:             {result.split_period}")
    print(f"  Bars processed:     {result.n_bars}")
    print(f"  Total trades:       {m.get('total_trades', 0)}")
    print(f"  Annualized Return: {m.get('annualized_return', 0):.2f}%")
    print(f"  Sharpe Ratio:       {m.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino Ratio:      {m.get('sortino_ratio', 0):.3f}")
    print(f"  Max Drawdown:       {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  CVaR (95%):         {m.get('cvar_95', 0):.3f}%")
    print(f"  Win Rate:           {m.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor:      {m.get('profit_factor', 0):.3f}")
    print(f"  Calmar Ratio:       {m.get('calmar_ratio', 0):.3f}")
    print("=" * 60)

    # Save equity to CSV
    import pandas as pd
    eq_df = pd.DataFrame({
        "bar":        list(range(len(result.equity_curve))),
        "equity":     result.equity_curve,
        "drawdown":   result.drawdown_curve,
    })
    out_path = project_root / "reports" / "backtest_result.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eq_df.to_csv(out_path, index=False)
    print(f"\n[Saved] equity curve -> {out_path}")


if __name__ == "__main__":
    main()
