"""
RSI Mean-Reversion Strategy

RSI (Relative Strength Index) measures momentum on a 0–100 scale.
Mean-reversion logic:
  - BUY when RSI drops below oversold threshold (default 30) — asset may be oversold
  - SELL when RSI rises above overbought threshold (default 70) — asset may be overbought

This is a contrarian strategy, opposite to trend-following. It performs
differently from SMA crossover, making it useful for comparison in the tearsheet.

Position sizing: same pattern as SMA strategy — invest a fixed % of equity.

Usage:
  from src.strategies.rsi import RSIStrategy
  strategy = RSIStrategy("SPY", period=14, oversold=30, overbought=70)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.strategy import BaseStrategy
from src.execution import Order
from src.portfolio import Portfolio


def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EWM with alpha = 1/period).
    This matches the standard RSI definition from Welles Wilder (1978).
    """
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


class RSIStrategy(BaseStrategy):
    """
    RSI mean-reversion strategy.

    Args:
        ticker: Ticker symbol to trade
        period: RSI lookback period (default 14 days — Wilder's original)
        oversold: RSI level below which we consider the asset oversold (default 30)
        overbought: RSI level above which we consider the asset overbought (default 70)
        position_pct: Fraction of equity to invest when entering (default 0.95)
    """

    def __init__(
        self,
        ticker: str,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        position_pct: float = 0.95,
    ):
        super().__init__(ticker)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.position_pct = position_pct
        self._rsi: pd.Series | None = None

    def on_start(self, data: pd.DataFrame):
        self._rsi = compute_rsi(data["Close"], self.period)

    def on_bar(
        self,
        date: pd.Timestamp,
        data: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[Order]:
        if len(data) < self.period + 1:
            return []

        if self._rsi is None:
            self.on_start(data)

        try:
            rsi_today = self._rsi.loc[date]
        except KeyError:
            return []

        if np.isnan(rsi_today):
            return []

        price = data.loc[date, "Close"]
        position = portfolio.get_position(self.ticker)

        orders = []

        # Oversold → potential reversal up → BUY
        if rsi_today < self.oversold and position == 0:
            equity = portfolio.cash
            target_value = equity * self.position_pct
            quantity = int(target_value / price)
            if quantity > 0:
                orders.append(Order(self.ticker, "BUY", quantity))

        # Overbought → potential reversal down → SELL
        elif rsi_today > self.overbought and position > 0:
            orders.append(Order(self.ticker, "SELL", position))

        return orders
