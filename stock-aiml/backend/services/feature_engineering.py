"""Feature engineering for OHLCV time-series data.

All features are computed using only PAST data (no lookahead).
Uses only closed candle values to compute features at bar t.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import zscore

if TYPE_CHECKING:
    pass


# ─── Scalar helpers ────────────────────────────────────────────────────────────

def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast,  adjust=False).mean()
    ema_slow = close.ewm(span=slow,  adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma    = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = _true_range(high, low, prev_close)
    return tr.rolling(window).mean()


# ─── Main FeatureEngine ────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Computes engineered features for each bar of OHLCV data.

    All features are strictly non-repaint: features at time t use only
    data up to and including time t (no future peeking).
    """

    def __init__(
        self,
        return_horizons: list[int] = [1, 5, 10, 20],
        vol_windows:     list[int] = [5, 10, 20],
        momentum_windows: list[int] = [5, 10, 20],
        vol_window:      int = 20,
        atr_window:      int = 14,
    ):
        self.return_horizons  = return_horizons
        self.vol_windows      = vol_windows
        self.momentum_windows = momentum_windows
        self.vol_window       = vol_window
        self.atr_window       = atr_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered columns to df IN-PLACE and return the same df.

        No new columns are added to the input — a copy is returned.
        All lookups are backward-only (shift(0) is NEVER used for future values).
        """
        df = df.copy()
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        vol   = df["volume"]

        # ── Returns at multiple horizons ──────────────────────────────────
        for h in self.return_horizons:
            df[f"return_{h}d"] = np.log(close / close.shift(h))
            df[f"return_{h}d"] = df[f"return_{h}d"].fillna(0)

        # ── Rolling volatility ─────────────────────────────────────────────
        for w in self.vol_windows:
            df[f"vol_{w}d"] = df[f"return_{1}d" if 1 in self.return_horizons else "return_1d"].rolling(w).std()
        # alias for the primary vol feature
        if 20 in self.vol_windows:
            df["volatility"] = df["vol_20d"]

        # ── Momentum indicators ────────────────────────────────────────────
        for w in self.momentum_windows:
            df[f"momentum_{w}d"] = close / close.shift(w) - 1

        # ── RSI ─────────────────────────────────────────────────────────────
        df["rsi_14"] = _rsi(close, window=14)
        df["rsi_7"]  = _rsi(close, window=7)

        # ── MACD ───────────────────────────────────────────────────────────
        macd, signal, hist = _macd(close)
        df["macd"]      = macd
        df["macd_signal"] = signal
        df["macd_hist"]  = hist

        # ── Bollinger Bands ────────────────────────────────────────────────
        bb_upper, bb_mid, bb_lower = _bollinger_bands(close)
        df["bb_upper"]   = bb_upper
        df["bb_mid"]     = bb_mid
        df["bb_lower"]   = bb_lower
        df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # ── ATR ────────────────────────────────────────────────────────────
        df["atr"]         = _atr(high, low, close, window=self.atr_window)
        df["atr_pct"]    = df["atr"] / close * 100  # ATR as % of price

        # ── Volume features ─────────────────────────────────────────────────
        df["volume_sma20"]     = vol.rolling(self.vol_window).mean()
        df["volume_ratio"]     = vol / df["volume_sma20"]
        df["volume_change"]   = vol.pct_change()
        df["volume_change"]   = df["volume_change"].clip(-5, 5).fillna(0)

        # ── Price-based features ──────────────────────────────────────────
        df["high_low_range"]  = (high - low) / close
        df["close_open_gap"]  = (close - df["open"]) / df["open"]
        df["upper_shadow"]    = (high - pd.concat([high, close], axis=1).max(axis=1)) / close
        df["lower_shadow"]    = (pd.concat([low, close], axis=1).min(axis=1) - low) / close

        # ── Z-score of return (for regime detection) ───────────────────────
        df["return_zscore"] = zscore(df["return_1d"].fillna(0), nan_policy="omit")

        # ── Regime flag: 1=trending up, -1=trending down, 0=neutral ────────
        df["regime"] = 0
        rising  = (df["macd"] > df["macd_signal"]) & (df["rsi_14"] > 50)
        falling = (df["macd"] < df["macd_signal"]) & (df["rsi_14"] < 50)
        df.loc[rising, "regime"] =  1
        df.loc[falling, "regime"] = -1

        return df

    def get_feature_columns(self, df: pd.DataFrame | None = None) -> list[str]:
        """Return the list of engineered feature columns (not raw OHLCV)."""
        candidate = [
            "return_1d", "return_5d", "return_10d", "return_20d",
            "vol_5d", "vol_10d", "vol_20d",
            "momentum_5d", "momentum_10d", "momentum_20d",
            "rsi_14", "rsi_7",
            "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_mid", "bb_lower", "bb_position",
            "atr", "atr_pct",
            "volume_sma20", "volume_ratio", "volume_change",
            "high_low_range", "close_open_gap",
            "return_zscore", "regime",
        ]
        if df is not None:
            return [c for c in candidate if c in df.columns]
        return candidate


# ─── Cross-asset features (for TGNN) ─────────────────────────────────────────

def build_cross_asset_features(
    multi_df: dict[str, pd.DataFrame],
    window: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Build cross-asset features for multiple stock DataFrames.

    Args:
        multi_df: dict mapping stock_symbol -> DataFrame (must have same index range)
        window: lookback window for rolling correlations

    Returns:
        dict mapping stock_symbol -> DataFrame with cross-asset columns added
    """
    symbols = list(multi_df.keys())
    combined = pd.concat([df[["close", "volume"]].rename(
        columns={"close": f"{s}_close", "volume": f"{s}_volume"}
    ) for s, df in multi_df.items()], axis=1)

    results = {}
    for sym in symbols:
        df = multi_df[sym].copy()
        close_col = f"{sym}_close"

        # Rolling cross-correlations with other stocks
        for other in symbols:
            if other == sym:
                continue
            other_col = f"{other}_close"
            if other_col not in combined.columns:
                continue
            ret_sym  = df["close"].pct_change().fillna(0)
            ret_oth  = combined[other_col].pct_change().fillna(0)
            corr = ret_sym.rolling(window).corr(ret_oth)
            df[f"corr_{other}_{window}d"] = corr.fillna(0)

        # Lagged cross-correlations (lead/lag signal)
        for lag in [1, 2, 5]:
            lead_ret = combined[f"{sym}_close"].pct_change().shift(-lag).fillna(0)
            for other in symbols:
                if other == sym:
                    continue
                other_col = f"{other}_close"
                if other_col not in combined.columns:
                    continue
                ret_oth = combined[other_col].pct_change().fillna(0)
                df[f"lead_{lag}d_corr_{other}"] = lead_ret.rolling(window).corr(ret_oth).fillna(0)

        results[sym] = df

    return results
