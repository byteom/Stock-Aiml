"""
RL Optimizer Service — Connects PPO Agent to the Backtester
============================================================
Wraps the trained PPO model and exposes optimize() for the API.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d",
    "vol_5d", "vol_10d", "vol_20d",
    "rsi", "macd", "macd_sig", "bb_pos", "atr", "volume_ratio",
    "hl_range",
]

# 4 discrete strategy parameter choices for the RL agent
PARAM_CHOICES = [
    (10, 0.005, 0.5),  # 0: conservative short-term
    (20, 0.01,  1.0),  # 1: standard momentum
    (20, 0.02,  1.0),  # 2: aggressive momentum
    (30, 0.01,  1.0),  # 3: high conviction trend
]


class RLOptimizer:
    """
    RL-based strategy parameter optimizer.
    Loads a trained PPO policy and selects best parameters for a given market window.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = model_path or "stock-aiml/models/rl/best_ppo.pt"
        self.policy = None
        self._load()

    def _load(self):
        # Try multiple possible paths for the RL model
        possible_paths = [
            self.model_path,
            Path("stock-aiml/models/rl/best_ppo.pt"),
            Path("models/rl/best_ppo.pt"),
        ]
        found_path = None
        for p in possible_paths:
            if Path(p).exists():
                found_path = p
                break

        if found_path is None:
            print(f"[RLOptimizer] No RL model found at any path, using grid search fallback")
            self.policy = None
            return

        ckpt = torch.load(found_path, map_location=DEVICE, weights_only=False)
        from ml.rl.train_rl import ActorCritic
        config = ckpt["config"]
        self.policy = ActorCritic(
            config["state_dim"], config["num_actions"], config["hidden_dim"]
        ).to(DEVICE)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.policy.eval()
        print(f"[RLOptimizer] Loaded PPO policy from {self.model_path}")

    def _build_state(
        self,
        df: pd.DataFrame,
        end_idx: int,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        equity: float,
        initial_capital: float,
        pos_size: float,
    ) -> np.ndarray:
        """Build 302-dim state vector."""
        lookback = 20
        feat_arr = df[FEATURE_COLS].values
        window = feat_arr[max(0, end_idx - lookback):end_idx]
        if len(window) < lookback:
            pad = np.zeros((lookback - len(window), window.shape[1]))
            window = np.vstack([pad, window])
        feat = (window - feat_mean) / feat_std
        equity_ratio = equity / initial_capital
        state = np.concatenate([feat.flatten(), [equity_ratio - 1.0], [pos_size]])
        return state.astype(np.float32)

    def select_action(self, state: np.ndarray) -> int:
        """Run policy to get best action for current state."""
        if self.policy is None:
            return 1  # fallback: standard momentum
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
            action, _, _ = self.policy.get_action(s)
        return action.item()

    def grid_search(
        self,
        df: pd.DataFrame,
        strategy_name: str = "momentum",
        param_grid: dict[str, list] | None = None,
        initial_capital: float = 100_000.0,
    ) -> dict[str, Any]:
        """
        Exhaustively search parameter grid, running backtest for each combo.
        Returns best params and all trial results.
        """
        from backend.services.backtester import Backtester
        from backend.services.execution_model import ExecutionSurrogate

        if param_grid is None:
            param_grid = {
                "lookback_short": [3, 5, 7],
                "lookback_long":  [15, 20, 25],
                "stop_loss_pct":  [0.01, 0.02, 0.03],
                "take_profit_pct": [0.03, 0.05, 0.07],
                "max_position_pct": [0.5, 1.0],
            }

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        trials = []
        best_score = -9999
        best_params = {}
        best_metrics = {}

        from itertools import product
        for combo in product(*values):
            params = dict(zip(keys, combo))

            exec_surrogate = ExecutionSurrogate(seed=42)
            bt = Backtester(
                initial_capital=initial_capital,
                commission_pct=0.001,
                slippage_bps=5.0,
                exec_surrogate=exec_surrogate,
            )

            try:
                result = bt.run(
                    df=df,
                    strategy_name=strategy_name,
                    strategy_params=params,
                    stop_loss_pct=params.get("stop_loss_pct", 0.02),
                    take_profit_pct=params.get("take_profit_pct", 0.05),
                    stochastic_exec=False,
                )
                score = result.metrics.get("sharpe_ratio", 0) * 100 - result.metrics.get("max_drawdown", 0)
            except Exception:
                score = -9999
                result = None

            trial = {"params": params, "score": score}
            if result:
                trial["metrics"] = result.metrics
                trial["total_trades"] = result.metrics.get("total_trades", 0)
                trial["annualized_return"] = result.metrics.get("annualized_return", 0)
                trial["sharpe_ratio"] = result.metrics.get("sharpe_ratio", 0)
                trial["max_drawdown_pct"] = result.metrics.get("max_drawdown_pct", 0)

            trials.append(trial)

            if score > best_score:
                best_score = score
                best_params = params
                if result:
                    best_metrics = result.metrics

        trials.sort(key=lambda x: x["score"], reverse=True)

        class GridResult:
            def __init__(self, bp, bm, trials):
                self.best_params = bp
                self.best_metrics = bm
                self.all_param_trials = trials

        return GridResult(best_params, best_metrics, trials)

    def optimize(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
    ) -> dict[str, Any]:
        """
        Optimize strategy parameters using trained PPO agent.
        Returns recommended params + walk-forward performance.
        """
        n = len(df)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        # Normalize on train
        train_df = df.iloc[:train_end]
        feat_mean = train_df[FEATURE_COLS].mean().values
        feat_std  = train_df[FEATURE_COLS].std().values + 1e-8

        equity = initial_capital
        equity_curve = [equity]
        trades = []

        for i in range(train_end, val_end):
            state = self._build_state(
                df, i, feat_mean, feat_std, equity, initial_capital, 1.0
            )
            action = self.select_action(state)
            lookback, threshold, pos_size = PARAM_CHOICES[action]

            ret_lookback = float(df["ret_1d"].iloc[max(0, i - lookback):i].sum())
            vol          = float(df["vol_5d"].iloc[i - 1]) if i > 0 else 0.01
            ret_today    = float(df["ret_1d"].iloc[i])

            if ret_lookback > threshold * lookback:
                direction = 1
            elif ret_lookback < -threshold * lookback:
                direction = -1
            else:
                direction = 0

            pnl      = equity * pos_size * direction * ret_today
            slippage = equity * pos_size * vol * 0.001
            equity   = max(1.0, equity + pnl - slippage)
            equity_curve.append(equity)

            if direction != 0:
                trades.append({"idx": i, "direction": direction, "equity": equity})

        from backend.services.metrics import compute_metrics, metrics_to_dict
        metrics = compute_metrics(equity_curve)
        metrics_dict = metrics_to_dict(metrics)

        # Recommend params at end of validation period
        best_action = self.select_action(
            self._build_state(
                df, val_end, feat_mean, feat_std, equity, initial_capital, 1.0
            )
        )
        best_params = PARAM_CHOICES[best_action]

        return {
            "recommended_params": {
                "lookback":  best_params[0],
                "threshold": best_params[1],
                "pos_size":  best_params[2],
            },
            "agent_action": best_action,
            "equity_curve": [round(e, 2) for e in equity_curve],
            "metrics": metrics_dict,
            "num_trades": len(trades),
        }
