"""Adversarial Market Generator (AMG) — generates plausible worst-case scenarios.

For MVP: rule-based worst-case scenario injection
  - Generates scenarios by modifying historical data with plausible shocks
  - Maintains temporal consistency (momentum, volatility clustering)
  - Scenario types: liquidity_shock, correlated_selloff, volatility_spike,
    regime_shift, fast_reversal

For full implementation: conditional GAN (see generator.py / discriminator.py)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


# ─── Scenario types ────────────────────────────────────────────────────────────

class ScenarioType(str, Enum):
    LIQUIDITY_SHOCK    = "liquidity_shock"
    CORRELATED_SELLOFF = "correlated_selloff"
    VOLATILITY_SPIKE   = "volatility_spike"
    REGIME_SHIFT       = "regime_shift"
    FAST_REVERSAL      = "fast_reversal"
    CRASH              = "crash"


@dataclass
class StressScenario:
    """A single generated adversarial scenario."""
    name:        str
    scenario_type: ScenarioType
    start_bar:   int
    end_bar:     int
    magnitude:   float          # e.g. -0.10 = 10% drop
    vol_mult:    float = 1.0    # volatility multiplier
    volume_mult: float = 1.0    # volume multiplier
    generated_df: pd.DataFrame | None = None  # modified data
    plausibility_score: float = 0.7


# ─── Scenario injection functions ──────────────────────────────────────────────

def _inject_liquidity_shock(
    df: pd.DataFrame,
    start_bar: int,
    end_bar: int,
    magnitude: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Inject a liquidity shock: volume drops sharply, spreads widen.
    Prices barely move (low volume doesn't move markets much in this model).
    """
    result = df.copy()
    shock_bars = result.iloc[start_bar:end_bar]
    result.loc[shock_bars.index, "volume"] = (
        shock_bars["volume"] * (1 - magnitude)
    ).clip(lower=1.0)
    return result


def _inject_correlated_selloff(
    df: pd.DataFrame,
    start_bar: int,
    end_bar: int,
    magnitude: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Inject a correlated selloff: smooth drop in close prices.
    Uses a sine-modulated decay to avoid discontinuities.
    """
    result = df.copy()
    n = end_bar - start_bar
    t = np.arange(n)
    # Smooth drop profile: start fast, slow near bottom
    drop_profile = magnitude * (1 - np.exp(-t / max(n / 3, 1)))
    # magnitude < 0 (e.g., -0.10 = 10% drop), so price_scale < 1
    price_scale = 1 + drop_profile
    for i, idx in enumerate(result.index[start_bar:end_bar]):
        result.loc[idx, "close"] *= price_scale[i]
        result.loc[idx, "high"]  *= price_scale[i]
        result.loc[idx, "low"]   *= price_scale[i]
        result.loc[idx, "open"]   *= price_scale[i]
    return result


def _inject_volatility_spike(
    df: pd.DataFrame,
    start_bar: int,
    end_bar: int,
    magnitude: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Inject a volatility spike: increase intra-bar range (high-low spread)
    without changing the close direction significantly.
    """
    result = df.copy()
    for idx in result.index[start_bar:end_bar]:
        row = result.loc[idx]
        mid = (row["high"] + row["low"]) / 2
        base_range = row["high"] - row["low"]
        new_range = base_range * magnitude
        half_new  = new_range / 2
        result.loc[idx, "high"] = mid + half_new
        result.loc[idx, "low"]  = mid - half_new
        # Keep close in the expanded range
        new_close = rng.uniform(mid - half_new * 0.9, mid + half_new * 0.9)
        result.loc[idx, "close"] = new_close
    return result


def _inject_regime_shift(
    df: pd.DataFrame,
    start_bar: int,
    end_bar: int,
    magnitude: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Inject a persistent regime shift: all prices shift by a fraction.
    """
    result = df.copy()
    for col in ("open", "high", "low", "close"):
        result.loc[result.index[start_bar:end_bar], col] *= (1 + magnitude)
    return result


def _inject_fast_reversal(
    df: pd.DataFrame,
    start_bar: int,
    end_bar: int,
    magnitude: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Fast drop followed by quick recovery: V-shaped shock.
    """
    result = df.copy()
    n = end_bar - start_bar
    t = np.arange(n)
    # V-shape profile
    midpoint = n // 2
    drop_profile = np.where(
        t <= midpoint,
        -magnitude * t / midpoint,
        -magnitude * (n - t) / (n - midpoint),
    )
    price_scale = 1 + drop_profile
    for i, idx in enumerate(result.index[start_bar:end_bar]):
        for col in ("open", "high", "low", "close"):
            result.loc[idx, col] *= price_scale[i]
    return result


_INJECTORS = {
    ScenarioType.LIQUIDITY_SHOCK:    _inject_liquidity_shock,
    ScenarioType.CORRELATED_SELLOFF: _inject_correlated_selloff,
    ScenarioType.VOLATILITY_SPIKE:   _inject_volatility_spike,
    ScenarioType.REGIME_SHIFT:       _inject_regime_shift,
    ScenarioType.FAST_REVERSAL:      _inject_fast_reversal,
    ScenarioType.CRASH:              _inject_correlated_selloff,  # crash = severe selloff
}


# ─── Main generator ────────────────────────────────────────────────────────────

class AdversarialMarketGenerator:
    """
    Generates plausible worst-case market scenarios for stress testing.

    MVP approach: rule-based scenario injection.
    Full approach (planned): conditional GAN that learns from historical distributions.

    The generator:
      1. Selects a random scenario type
      2. Picks a plausible start/end window in the data
      3. Injects the shock with controlled magnitude
      4. Validates plausibility (bounds on max drawdown, volatility)
      5. Returns modified DataFrame for use in backtester
    """

    def __init__(
        self,
        scenarios: list[dict] | None = None,
        seed: int = 42,
    ):
        self.rng  = np.random.default_rng(seed)
        self.scenarios = scenarios or [
            {"type": "liquidity_shock",    "magnitude": 0.15, "duration": 15},
            {"type": "correlated_selloff", "magnitude": -0.10, "duration": 15},
            {"type": "volatility_spike",   "magnitude": 2.0, "duration": 10},
            {"type": "regime_shift",        "magnitude": -0.05, "duration": 30},
            {"type": "fast_reversal",       "magnitude": -0.08, "duration": 10},
        ]

    def generate_scenario(
        self,
        df: pd.DataFrame,
        scenario_type: ScenarioType | str | None = None,
        magnitude: float | None = None,
        duration: int | None = None,
        start_bar: int | None = None,
    ) -> StressScenario:
        """
        Generate a single adversarial scenario.

        Returns a StressScenario with the modified DataFrame.
        """
        # Normalize scenario_type: accept string values or ScenarioType enums
        if scenario_type is not None:
            stype_val = str(scenario_type).strip()
            # Strip any class prefix like "ScenarioType." or "ScenarioType."
            if "." in stype_val:
                stype_val = stype_val.rsplit(".", 1)[-1].strip()
            try:
                scenario_type = ScenarioType(stype_val)
            except (ValueError, TypeError):
                # Fallback: try to normalize common formats
                stype_val = stype_val.lower().replace(" ", "_").replace("-", "_")
                try:
                    scenario_type = ScenarioType(stype_val)
                except (ValueError, TypeError):
                    scenario_type = None

        # Pick scenario type if not specified
        if scenario_type is None:
            sc = self.rng.choice(self.scenarios)
        else:
            sc = next((s for s in self.scenarios if s["type"] == scenario_type.value), self.scenarios[0])

        sc_type  = ScenarioType(sc["type"])
        mag      = magnitude if magnitude is not None else float(sc["magnitude"])
        dur      = duration  if duration  is not None else int(sc["duration"])

        # Choose start bar (middle third of data to allow room for scenario)
        n = len(df)
        safe_start_min = int(n * 0.3)
        safe_start_max = int(n * 0.7)
        if start_bar is None:
            sb = self.rng.integers(safe_start_min, safe_start_max)
        else:
            sb = start_bar
        eb = min(sb + dur, n)

        # Clamp magnitude to plausible bounds
        if sc_type in (ScenarioType.CORRELATED_SELLOFF, ScenarioType.CRASH):
            mag = max(-0.30, min(mag, -0.01))   # negative means drop, bounded
        elif sc_type == ScenarioType.LIQUIDITY_SHOCK:
            mag = max(0.05, min(mag, 0.40))
        elif sc_type == ScenarioType.VOLATILITY_SPIKE:
            mag = max(1.2, min(mag, 4.0))

        # Inject the scenario
        injector = _INJECTORS.get(sc_type)
        if injector:
            modified_df = injector(df, sb, eb, mag, self.rng)
        else:
            modified_df = df

        # Compute plausibility score (simple heuristic: within reasonable bounds)
        if sc_type == ScenarioType.CORRELATED_SELLOFF:
            price_change = (modified_df["close"].iloc[eb-1] / modified_df["close"].iloc[sb] - 1)
            plausibility = float(np.clip(1 - abs(price_change) * 2, 0, 1))
        elif sc_type == ScenarioType.VOLATILITY_SPIKE:
            vol_ratio = (modified_df["high"] - modified_df["low"]).mean() / \
                        (df["high"] - df["low"]).mean()
            plausibility = float(np.clip(1 - (vol_ratio - 1) * 0.5, 0, 1))
        else:
            plausibility = 0.75

        return StressScenario(
            name=f"{sc_type.value}_{sb}_{eb}",
            scenario_type=sc_type,
            start_bar=sb,
            end_bar=eb,
            magnitude=mag,
            vol_mult=mag if sc_type == ScenarioType.VOLATILITY_SPIKE else 1.0,
            volume_mult=(1 - mag) if sc_type == ScenarioType.LIQUIDITY_SHOCK else 1.0,
            generated_df=modified_df,
            plausibility_score=plausibility,
        )

    def generate_n_scenarios(
        self,
        df: pd.DataFrame,
        n: int = 50,
        scenario_types: list[ScenarioType | str] | None = None,
    ) -> list[StressScenario]:
        """Generate N adversarial scenarios."""
        scenarios = []
        # Normalize: keep only ScenarioType objects, not strings that are already enum values
        types = []
        for t in (scenario_types or []):
            if isinstance(t, ScenarioType):
                types.append(t)
            elif isinstance(t, str):
                try:
                    types.append(ScenarioType(t))
                except (ValueError, TypeError):
                    pass
        for _ in range(n):
            st = self.rng.choice(types) if types else None
            sc = self.generate_scenario(df, scenario_type=st)
            scenarios.append(sc)
        return scenarios

    def get_scenario_description(self, sc: StressScenario) -> str:
        """Human-readable description of a scenario."""
        type_desc = {
            ScenarioType.LIQUIDITY_SHOCK: "a sharp volume drop",
            ScenarioType.CORRELATED_SELLOFF: "a correlated market selloff",
            ScenarioType.VOLATILITY_SPIKE: "a sudden volatility spike",
            ScenarioType.REGIME_SHIFT: "a persistent regime shift",
            ScenarioType.FAST_REVERSAL: "a fast V-shaped reversal",
            ScenarioType.CRASH: "a sudden market crash",
        }
        desc = type_desc.get(sc.scenario_type, "an unknown shock")
        return (f"{desc.capitalize()} starting at bar {sc.start_bar} "
                f"lasting {sc.end_bar - sc.start_bar} bars with "
                f"magnitude {sc.magnitude:.2%} (plausibility: {sc.plausibility_score:.0%})")
