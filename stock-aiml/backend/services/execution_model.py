"""Execution surrogate — models fill probability, market impact, and slippage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class FillResult(BaseModel):
    """Result of attempting to execute an order."""
    filled:      bool              # whether the order was filled
    fill_price:  float             # price at which it was filled
    slippage_bps: float = 0.0      # slippage in basis points vs signal price
    market_impact_bps: float = 0.0 # adverse market impact in bps
    total_cost_bps: float = 0.0    # total execution cost in bps
    fill_prob:   float = 1.0       # fill probability (0–1)


class ExecutionSurrogate:
    """
    Parametric execution model for simulating realistic fills.

    No execution logs are available for this project, so we use a
    calibrated parametric model based on the literature:

        impact = C * (order_size / ADV) ** alpha * sigma * spread_factor

    Parameters are configurable via config.yaml.
    """

    def __init__(
        self,
        impact_coeff:     float = 0.5,
        fill_rate:        float = 0.85,
        vol_impact_mult: float = 1.2,
        spread_factor:   float = 0.5,
        seed:            int = 42,
    ):
        self.impact_coeff    = impact_coeff
        self.fill_rate       = fill_rate
        self.vol_impact_mult = vol_impact_mult
        self.spread_factor   = spread_factor
        self.rng             = np.random.default_rng(seed)

    def simulate_fill(
        self,
        signal_price:  float,       # price at which signal fires (decision price)
        order_size:    float,       # number of units to trade
        direction:     int,         # +1 long, -1 short
        current_vol:   float,       # recent volatility (std of returns)
        adv:           float,       # average daily volume (units)
        spread_bps:    float = 5.0, # bid-ask spread in bps
        latency_ms:    float = 100.0, # simulated execution latency (ms)
        stochastic:    bool = True, # add random fill/no-fill draw
    ) -> FillResult:
        """
        Simulate the execution of a single order.

        Returns a FillResult with fill status, executed price, and cost breakdown.
        """
        # ── 1. Fill probability ──────────────────────────────────────────────
        if order_size <= 0:
            return FillResult(filled=False, fill_price=signal_price, fill_prob=0.0)

        participation_rate = order_size / adv if adv > 0 else order_size
        fill_prob = self.fill_rate * np.exp(-participation_rate * 2)
        fill_prob = float(np.clip(fill_prob, 0.0, 1.0))

        if stochastic:
            filled = self.rng.random() < fill_prob
        else:
            filled = fill_prob > 0.5

        if not filled:
            return FillResult(
                filled=False,
                fill_price=signal_price,
                slippage_bps=0.0,
                market_impact_bps=0.0,
                total_cost_bps=0.0,
                fill_prob=fill_prob,
            )

        # ── 2. Slippage ──────────────────────────────────────────────────────
        # Slippage increases with latency and volatility
        latency_factor = np.sqrt(latency_ms / 1000.0)  # sqrt scales with time
        slippage_bps = spread_bps * (0.5 + 0.5 * latency_factor)
        slippage_bps = float(slippage_bps)

        # ── 3. Market impact ───────────────────────────────────────────────
        # Impact scales with order size relative to ADV and volatility
        impact = self.impact_coeff * participation_rate * self.vol_impact_mult * current_vol
        market_impact_bps = float(impact * 1e4)  # convert fraction to bps

        # ── 4. Total execution cost ───────────────────────────────────────
        # spread cost + impact (always pay spread; impact depends on direction)
        spread_cost_bps = spread_bps * self.spread_factor
        total_cost_bps = spread_cost_bps + market_impact_bps

        # Adjust fill price: adverse movement for taker
        # BUY: price goes up by cost; SELL: price goes down by cost
        fill_adj = direction * total_cost_bps / 10000.0
        fill_price = signal_price * (1.0 + fill_adj)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            total_cost_bps=total_cost_bps,
            fill_prob=fill_prob,
        )

    def compute_transaction_cost(
        self,
        order_value: float,
        commission_pct: float = 0.001,
        slippage_bps: float = 5.0,
    ) -> float:
        """Calculate total transaction cost for an order."""
        commission_cost = order_value * commission_pct * 2  # both entry and exit
        slippage_cost   = order_value * slippage_bps / 10000.0
        return commission_cost + slippage_cost

    def estimate_fill_rate_curve(
        self,
        adv: float,
        vol: float,
        max_size_factor: float = 2.0,
    ) -> pd.DataFrame:
        """
        Return a fill-rate curve as a function of order size.
        Useful for showing traders the impact of large order sizes.
        """
        sizes = np.linspace(0.01, max_size_factor * adv, 100)
        fill_probs = [self.fill_rate * np.exp(-(s / adv) * 2) for s in sizes]
        return pd.DataFrame({"order_size": sizes, "fill_prob": fill_probs}).clip(
            0, 1
        )
