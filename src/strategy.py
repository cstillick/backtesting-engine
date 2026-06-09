"""
Base strategy interface for the backtesting engine.

All strategies inherit from BaseStrategy and implement on_bar().
The engine calls on_bar() once per trading day with the current bar data
and the current portfolio state.

Design philosophy:
- Strategies are stateful — they can store indicators, position state, etc.
- Strategies return a list of Orders (can be empty)
- Strategies never modify the portfolio directly — that's the engine's job
- on_bar receives a "window" of historical data up to and including today,
  so strategies can compute any indicator they need

Example:
    class MyStrategy(BaseStrategy):
        def on_bar(self, date, data, portfolio):
            if some_signal:
                return [Order("AAPL", "BUY", 10)]
            return []
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from src.portfolio import Portfolio
    from src.execution import Order


class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.

    Subclasses implement on_bar() to define trading logic.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker

    def on_start(self, data: pd.DataFrame):
        """
        Called once before the backtest begins.
        Override to precompute indicators on the full price history.
        data: full OHLCV DataFrame (not just the window)
        """
        pass

    @abstractmethod
    def on_bar(
        self,
        date: pd.Timestamp,
        data: pd.DataFrame,
        portfolio: "Portfolio",
    ) -> list["Order"]:
        """
        Called once per trading bar (day).

        Args:
            date: Current bar's timestamp
            data: Historical OHLCV data up to and including today
            portfolio: Current portfolio state (read-only in strategy logic)

        Returns:
            List of Order objects (empty list = no action)
        """
        raise NotImplementedError

    def on_end(self, portfolio: "Portfolio"):
        """Called once after the backtest completes. Override for cleanup."""
        pass
