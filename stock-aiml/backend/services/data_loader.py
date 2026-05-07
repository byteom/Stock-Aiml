"""Data ingestion and preprocessing for OHLCV stock/index data."""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator

from backend.core.exceptions import DataNotFoundError, InvalidDateRangeError, InsufficientDataError


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class OHLCVSchema(BaseModel):
    """Expected schema for a single OHLCV row."""
    model_config = {"arbitrary_types_allowed": True}

    timestamp: str | pd.Timestamp
    open:      float = Field(..., gt=0)
    high:      float = Field(..., gt=0)
    low:       float = Field(..., gt=0)
    close:     float = Field(..., gt=0)
    volume:    float = Field(..., ge=0)
    # optional
    index_name: str | None = None
    spread:    float | None = None
    tick_count: int | None = None


# ─── Core loader ───────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads and validates OHLCV CSV data.

    Handles the NIFTY 50 format:
        "Index Name","Date","Open","High","Low","Close"
        "NIFTY 50","02 Jul 2025","25588.3","25608.1","25378.75","25453.40"

    The data.csv only has Open/High/Low/Close — no Volume column.
    We synthesize a volume proxy from the intra-bar range (high-low).
    """

    DATE_FORMATS = [
        "%d %b %Y",      # "02 Jul 2025"
        "%Y-%m-%d",      # "2025-07-02"
        "%d/%m/%Y",      # "02/07/2025"
        "%Y/%m/%d",      # "2025/07/02"
        "%d-%m-%Y",      # "02-07-2025"
    ]

    def __init__(self, data_path: str | Path | None = None):
        self.data_path = Path(data_path) if data_path else None
        self._raw_df: pd.DataFrame | None = None
        self._processed_df: pd.DataFrame | None = None

    # ── CSV loading ──────────────────────────────────────────────────────────

    def _synthesize_volume(self, df: pd.DataFrame) -> pd.Series:
        range_proxy = (df["high"] - df["low"]).abs()
        return (range_proxy / range_proxy.median() * 1e6).fillna(1e6).astype(float)

    def load_preset_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load data from an existing DataFrame directly.
        """
        self._raw_df = df.copy()
        
        # Standardize empty headers
        df.columns = df.columns.str.strip().str.lower()
        
        # Process and validate
        df = self._standardize_columns(df)
        self._processed_df = df
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check columns
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            # If missing raw columns, they might have capital letters
            col_map = {c: c.lower() for c in df.columns}
            df.rename(columns=col_map, inplace=True)
            if not required.issubset(set(df.columns)):
                raise ValueError(f"Missing required price columns: {required - set(df.columns)}")

        # Find date column
        date_cols = [c for c in df.columns if c in ["date", "timestamp", "time", "datetime"]]
        date_col = date_cols[0] if date_cols else df.columns[0]
        
        # Ensure timestamp exists
        if "timestamp" not in df.columns:
            df["timestamp"] = df[date_col]
            
        df["timestamp"] = self._parse_dates(df["timestamp"])
        df = df.sort_values("timestamp")
        df = df.reset_index(drop=True)
        
        # Generate volume if missing
        if "volume" not in df.columns:
            df["volume"] = self._synthesize_volume(df)
            
        return df

    def load_csv(
        self,
        path: str | Path | None = None,
        index_col: str | None = None,
        expected_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load a CSV with flexible column detection.
        Handles NIFTY 50 format: Index Name, Date, Open, High, Low, Close
        """
        path = Path(path) if path else self.data_path
        if not path or not path.exists():
            raise DataNotFoundError(f"Data file not found: {path}")

        # Detect delimiter and header
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        # Rename columns if needed (lowercase, strip whitespace)
        df.columns = [c.strip().lower() for c in df.columns]

        # Map common column names
        col_map = {}
        for col in df.columns:
            if col in ("index name", "index_name", "symbol", "ticker"):
                col_map[col] = "index_name"
            elif col in ("date", "timestamp", "datetime", "time"):
                col_map[col] = "timestamp"
            elif col in ("open", "o"):
                col_map[col] = "open"
            elif col in ("high", "h"):
                col_map[col] = "high"
            elif col in ("low", "l"):
                col_map[col] = "low"
            elif col in ("close", "c"):
                col_map[col] = "close"
            elif col in ("volume", "vol", "v"):
                col_map[col] = "volume"
            elif col in ("spread", "bid_ask_spread"):
                col_map[col] = "spread"
            elif col in ("trade_count", "tick_count"):
                col_map[col] = "tick_count"

        df = df.rename(columns=col_map)

        # Parse dates
        if "timestamp" in df.columns:
            df["timestamp"] = self._parse_dates(df["timestamp"])
        elif "date" in df.columns:
            df["timestamp"] = self._parse_dates(df["date"])
            df = df.drop(columns=["date"])
        else:
            raise DataNotFoundError("No date/timestamp column found in CSV")

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Synthesize volume if missing (proxy: range * constant)
        if "volume" not in df.columns:
            range_proxy = (df["high"] - df["low"]).abs()
            # Normalise to a reasonable volume-like scale
            df["volume"] = (range_proxy / range_proxy.median() * 1e6).fillna(1e6).astype(float)
            warnings.warn("No volume column found — synthesising volume proxy from intra-bar range")

        # Require OHLC
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise DataNotFoundError(f"Required column '{col}' not found in {path}")

        # Validate price consistency
        invalid = df[df["high"] < df["low"]]
        if not invalid.empty:
            warnings.warn(f"{len(invalid)} rows with high < low — dropping")
            df = df[df["high"] >= df["low"]]

        invalid2 = df[(df["close"] > df["high"]) | (df["close"] < df["low"])]
        if not invalid2.empty:
            warnings.warn(f"{len(invalid2)} rows with close outside [low,high] — clipping")
            df["close"] = df["close"].clip(df["low"], df["high"])

        self._raw_df = df
        self._processed_df = df.copy()
        return df

    def _parse_dates(self, series: pd.Series) -> pd.DatetimeIndex:
        for fmt in self.DATE_FORMATS:
            try:
                return pd.to_datetime(series, format=fmt)
            except (ValueError, TypeError):
                continue
        # Fallback: infer
        return pd.to_datetime(series, dayfirst=True)

    @property
    def raw_df(self) -> pd.DataFrame:
        if self._raw_df is None:
            raise DataNotFoundError("No data loaded — call load_csv() first")
        return self._raw_df

    @property
    def processed_df(self) -> pd.DataFrame:
        if self._processed_df is None:
            raise DataNotFoundError("No data loaded — call load_csv() first")
        return self._processed_df

    # ── Walk-forward splitter ────────────────────────────────────────────────

    def create_walk_forward_splits(
        self,
        n_splits: int = 3,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        gap_bars: int = 5,
        min_train_bars: int = 100,
    ) -> list[dict]:
        """
        Create non-overlapping walk-forward train/val/test splits.

        Each split:
            train: [start .. train_end]
            gap:   [train_end+1 .. train_end+gap_bars]
            val:   [gap_end+1 .. val_end]
            gap2:  [val_end+1 .. val_end+gap_bars]
            test:  [test_start .. end]

        Returns a list of dicts with split info.
        """
        df = self.processed_df
        n = len(df)
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise InvalidDateRangeError(f"Split ratios must sum to 1.0, got {total}")

        if n < min_train_bars * (1 / train_ratio):
            raise InsufficientDataError(
                f"Need at least {int(min_train_bars / train_ratio)} bars for walk-forward, got {n}"
            )

        splits = []
        # Walk forward in steps of test_ratio
        step = int(n * test_ratio)
        if step < 1:
            step = 1

        train_end_frac = train_ratio
        val_end_frac   = train_ratio + val_ratio

        for i in range(n_splits):
            offset = i * step

            train_end    = int(n * train_end_frac) + offset
            val_end       = int(n * val_end_frac) + offset
            test_end_idx  = min(n - 1, val_end + step)

            if train_end < min_train_bars:
                continue
            if train_end >= val_end or val_end >= test_end_idx:
                break

            train_df = df.iloc[:train_end].copy()
            val_df   = df.iloc[train_end + gap_bars : val_end].copy()
            test_df  = df.iloc[val_end + gap_bars : test_end_idx].copy()

            splits.append({
                "split_id":    i,
                "train_start": train_df.index[0],
                "train_end":   train_end,
                "val_start":   train_end + gap_bars,
                "val_end":     val_end,
                "test_start":  val_end + gap_bars,
                "test_end":    test_end_idx,
                "train":       train_df,
                "val":         val_df,
                "test":        test_df,
                "train_period": (
                    f"{train_df['timestamp'].iloc[0].date()} — {train_df['timestamp'].iloc[-1].date()}"
                ),
                "val_period": (
                    f"{val_df['timestamp'].iloc[0].date()} — {val_df['timestamp'].iloc[-1].date()}"
                ),
                "test_period": (
                    f"{test_df['timestamp'].iloc[0].date()} — {test_df['timestamp'].iloc[-1].date()}"
                ),
            })

        return splits


# ─── Convenience factory function ─────────────────────────────────────────────

def load_data(path: str | Path | None = None, **kwargs) -> DataLoader:
    """One-liner to load and process a data CSV."""
    loader = DataLoader(path)
    if path:
        loader.load_csv(path, **kwargs)
    return loader
