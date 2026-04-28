"""Unit tests for the adversarial stress generator."""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from backend.services.stress_generator import (
    AdversarialMarketGenerator,
    ScenarioType,
    _inject_correlated_selloff,
)


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    base  = 25000.0
    close = [float(base + i * 10 + rng.normal() * 20) for i in range(n)]
    df = pd.DataFrame({
        "timestamp": dates,
        "open":   close,
        "high":   [c * 1.01 for c in close],
        "low":    [c * 0.99 for c in close],
        "close":  close,
        "volume": [1_000_000] * n,
    })
    return df


class TestAdversarialGenerator:
    def test_generate_scenario_returns_stress_scenario(self):
        df = _make_df(100)
        gen = AdversarialMarketGenerator(seed=42)
        sc = gen.generate_scenario(df)
        assert sc.name is not None
        assert sc.scenario_type in ScenarioType
        assert sc.start_bar < sc.end_bar

    def test_selloff_reduces_prices(self):
        # Use a FLAT base price so the selloff effect dominates the trend
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        flat_df = pd.DataFrame({
            "timestamp": dates,
            "open":   [25000.0] * 100,
            "high":   [25050.0] * 100,
            "low":    [24950.0] * 100,
            "close":  [25000.0] * 100,
            "volume": [1_000_000] * 100,
        })
        rng = np.random.default_rng(42)
        result = _inject_correlated_selloff(flat_df, 30, 40, -0.10, rng)

        price_before = float(flat_df["close"].iloc[30])
        price_after  = float(result["close"].iloc[39])
        # A -10% magnitude selloff over 10 bars should reduce price significantly
        assert price_after < price_before * 0.95, f"Selloff should reduce prices by ≥5%: before={price_before:.2f}, after={price_after:.2f}"

    def test_generate_n_scenarios_returns_correct_count(self):
        df = _make_df(100)
        gen = AdversarialMarketGenerator(seed=42)
        scenarios = gen.generate_n_scenarios(df, n=20)
        assert len(scenarios) == 20

    def test_scenario_types_are_valid(self):
        df = _make_df(100)
        gen = AdversarialMarketGenerator(seed=42)
        for _ in range(50):
            sc = gen.generate_scenario(df)
            assert sc.scenario_type in ScenarioType

    def test_plausibility_score_is_bounded(self):
        df = _make_df(100)
        gen = AdversarialMarketGenerator(seed=42)
        for _ in range(20):
            sc = gen.generate_scenario(df)
            assert 0.0 <= sc.plausibility_score <= 1.0

    def test_get_description_returns_string(self):
        gen = AdversarialMarketGenerator(seed=42)
        df = _make_df(100)
        sc = gen.generate_scenario(df, scenario_type=ScenarioType.VOLATILITY_SPIKE)
        desc = gen.get_scenario_description(sc)
        assert isinstance(desc, str)
        assert len(desc) > 10
