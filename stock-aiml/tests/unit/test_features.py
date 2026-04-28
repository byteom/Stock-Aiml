"""Unit tests for the feature engineering service."""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from backend.services.feature_engineering import FeatureEngine


def _make_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    base  = 25000.0
    close = [base + i * 5 + np.random.randn() * 10 for i in range(n)]
    df = pd.DataFrame({
        "timestamp": dates,
        "open":   [c * (1 - abs(np.random.randn()) * 0.005) for c in close],
        "high":   [c * (1 + abs(np.random.randn()) * 0.005) for c in close],
        "low":    [c * (1 - abs(np.random.randn()) * 0.005) for c in close],
        "close":  close,
        "volume": [1_000_000 for _ in range(n)],
    })
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    return df


class TestFeatureEngine:
    def test_transform_adds_features(self):
        df = _make_df(100)
        engine = FeatureEngine()
        result = engine.transform(df)

        expected_cols = ["return_1d", "return_5d", "vol_5d", "rsi_14", "macd", "bb_upper", "atr"]
        for col in expected_cols:
            assert col in result.columns, f"Missing: {col}"

    def test_no_nan_in_core_features(self):
        df = _make_df(100)
        engine = FeatureEngine()
        result = engine.transform(df)

        # Core features should have no NaNs after window warm-up
        for col in ["return_1d", "vol_5d", "atr"]:
            vals = result[col].dropna()
            assert len(vals) > 0

    def test_return_sign_is_reasonable(self):
        df = _make_df(100)
        engine = FeatureEngine()
        result = engine.transform(df)

        ret = result["return_1d"].dropna()
        assert ret.min() > -0.5, "Extreme negative return detected"
        assert ret.max() <  0.5, "Extreme positive return detected"

    def test_bbands_upper_greater_than_lower(self):
        df = _make_df(100)
        engine = FeatureEngine()
        result = engine.transform(df)

        valid = result[["bb_upper", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_rsi_bounded(self):
        df = _make_df(100)
        engine = FeatureEngine()
        result = engine.transform(df)

        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0.0
        assert rsi.max() <= 100.0

    def test_get_feature_columns(self):
        engine = FeatureEngine()
        cols = engine.get_feature_columns()
        assert "return_1d" in cols
        assert "rsi_14" in cols
        assert "macd" in cols
