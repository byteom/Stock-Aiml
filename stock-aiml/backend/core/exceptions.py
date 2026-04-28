"""Custom exceptions for the stock-aiml platform."""
from __future__ import annotations


class StockAIMLError(Exception):
    """Base exception for all stock-aiml errors."""
    pass


class DataNotFoundError(StockAIMLError):
    """Raised when required data files are missing."""
    pass


class InvalidDateRangeError(StockAIMLError):
    """Raised when date range is invalid (e.g., test before train)."""
    pass


class InsufficientDataError(StockAIMLError):
    """Raised when there isn't enough data for the requested operation."""
    pass


class StrategyConfigError(StockAIMLError):
    """Raised when strategy configuration is invalid."""
    pass


class BacktestError(StockAIMLError):
    """Raised when a backtest fails."""
    pass


class ModelNotFoundError(StockAIMLError):
    """Raised when a saved model checkpoint cannot be loaded."""
    pass


class WalkForwardError(StockAIMLError):
    """Raised when walk-forward split configuration is invalid."""
    pass
