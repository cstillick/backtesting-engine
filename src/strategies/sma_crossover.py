"""
SMA Crossover Strategy

Signal logic:
  - BUY when the fast SMA crosses above the slow SMA (golden cross)
  - SELL when the fast SMA crosses below the slow SMA (death cross)

This is one of the most classic technical strategies. It works reasonably
well in trending markets and is a clean way to demo the engine's core loop.

Position sizing: invest a fixed percentage of portfolio equity (default 95%)
to avoid being fully in cash. Integer shares only.

Usage:
  from src.strategies.sma_crossover import SMACrossover
  strategy = SMACrossover("AAPL", fast=20, slow=50)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.strategy import BaseStrategy
from src.execution import Order
from src.portfolio import Portfolio


class SMACrossover(BaseStrategy):
    """
    Simple Moving Average crossover strategy.

    Args:
        ticker: Ticker symbol to trade
        fast: Fast SMA window (default 20 days)
        slow: Slow SMA window (default 50 days)
        position_pct: Fraction of equity to invest when entering (default 0.95)
    """

    def __init__(
        self,
        ticker: str,
        fast: int = 20,
        slow: int = 50,
        position_pct: float = 0.95,
    ):
        super().__init__(ticker)
        self.fast = fast
        self.slow = slow
        self.position_pct = position_pct
        self._sma_fast: pd.Series | None = None
        self._sma_slow: pd.Series | None = None

    def on_start(self, data: pd.DataFrame):
        """Precompute SMAs on the full price history."""
        closes = data["Close"]
        self._sma_fast = closes.rolling(self.fast).mean()
        self._sma_slow = closes.rolling(self.slow).mean()

    def on_bar(
        self,
        date: pd.Timestamp,
        data: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[Order]:
        # Need at least slow+1 bars for a crossover signal
        if len(data) < self.slow + 1:
            return []

        if self._sma_fast is None or self._sma_slow is None:
            self.on_start(data)

        # Get today's and yesterday's values
        try:
            fast_today = self._sma_fast.loc[date]
            slow_today = self._sma_slow.loc[date]
            prev_date = data.index[data.index.get_loc(date) - 1]
            fast_prev = self._sma_fast.loc[prev_date]
            slow_prev = self._sma_slow.loc[prev_date]
        except (KeyError, IndexError):
            return []

        if any(np.isnan(v) for v in [fast_today, slow_today, fast_prev, slow_prev]):
            return []

        price = data.loc[date, "Close"]
        position = portfolio.get_position(self.ticker)

        orders = []

        # Golden cross: fast crosses above slow → BUY
        if fast_prev <= slow_prev and fast_today > slow_today and position == 0:
            equity = portfolio.cash  # simplified: use available cash
            target_value = equity * self.position_pct
            quantity = int(target_value / price)
            if quantity > 0:
                orders.append(Order(self.ticker, "BUY", quantity))

        # Death cross: fast crosses below slow → SELL all
        elif fast_prev >= slow_prev and fast_today < slow_today and position > 0:
            orders.append(Order(self.ticker, "SELL", position))

        return orders
