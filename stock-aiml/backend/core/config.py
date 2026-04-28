"""Configuration loader — reads config.yaml and strategy YAML files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Singleton config object loaded from config.yaml."""

    _instance: Config | None = None
    _data: dict[str, Any]

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        cfg_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
        with open(cfg_path, encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation access: config.get('backtester.initial_capital')"""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def section(self, section: str) -> dict[str, Any]:
        """Get an entire top-level section."""
        return self._data.get(section, {})

    @property
    def data(self) -> dict[str, Any]:
        return self._data


def load_strategy_config(strategy_name: str) -> dict[str, Any]:
    """Load a strategy configuration from configs/strategies/<name>.yaml"""
    strat_path = (
        Path(__file__).parent.parent / "configs" / "strategies" / f"{strategy_name}.yaml"
    )
    if not strat_path.exists():
        raise FileNotFoundError(f"Strategy config not found: {strat_path}")
    with open(strat_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# Global singleton
config = Config()
