"""
Portfolio state management for the backtesting engine.

Tracks:
- Cash balance
- Open positions (shares held per ticker)
- Equity curve (total portfolio value over time)
- Trade log (all fills)

This is a simplified but realistic model:
- No margin / shorting
- Integer share quantities
- Mark-to-market at each bar's close price
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Fill:
    """A completed order fill."""
    date: pd.Timestamp
    ticker: str
    side: str          # "BUY" or "SELL"
    quantity: int
    price: float       # fill price (after slippage)
    commission: float
    value: float       # quantity * price (before commission)


@dataclass
class Portfolio:
    """
    Manages cash, positions, and performance tracking.

    Args:
        initial_cash: Starting capital in USD
    """
    initial_cash: float = 100_000.0
    cash: float = field(init=False)
    positions: dict[str, int] = field(default_factory=dict)  # ticker → shares
    equity_curve: list[dict] = field(default_factory=list)    # [{date, equity}]
    trade_log: list[Fill] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_cash

    def get_position(self, ticker: str) -> int:
        return self.positions.get(ticker, 0)

    def apply_fill(self, fill: Fill):
        """
        Update cash and positions after a confirmed fill.
        BUY: deduct cash, add shares.
        SELL: add cash, remove shares.
        """
        if fill.side == "BUY":
            cost = fill.value + fill.commission
            if cost > self.cash:
                raise ValueError(
                    f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}"
                )
            self.cash -= cost
            self.positions[fill.ticker] = self.positions.get(fill.ticker, 0) + fill.quantity

        elif fill.side == "SELL":
            current = self.positions.get(fill.ticker, 0)
            if fill.quantity > current:
                raise ValueError(
                    f"Cannot sell {fill.quantity} shares of {fill.ticker}, only holding {current}"
                )
            self.cash += fill.value - fill.commission
            self.positions[fill.ticker] = current - fill.quantity
            if self.positions[fill.ticker] == 0:
                del self.positions[fill.ticker]

        self.trade_log.append(fill)

    def mark_to_market(self, date: pd.Timestamp, prices: dict[str, float]) -> float:
        """
        Compute total portfolio value at current prices and record to equity curve.
        prices: {ticker: close_price}
        """
        position_value = sum(
            shares * prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        )
        equity = self.cash + position_value
        self.equity_curve.append({"date": date, "equity": equity})
        return equity

    def get_equity_series(self) -> pd.Series:
        """Return equity curve as a DatetimeIndex Series."""
        df = pd.DataFrame(self.equity_curve)
        if df.empty:
            return pd.Series(dtype=float)
        return df.set_index("date")["equity"]

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "date": f.date,
                "ticker": f.ticker,
                "side": f.side,
                "quantity": f.quantity,
                "price": f.price,
                "commission": f.commission,
                "value": f.value,
            }
            for f in self.trade_log
        ])

    def __repr__(self) -> str:
        pos_str = ", ".join(f"{t}:{s}" for t, s in self.positions.items()) or "none"
        return f"Portfolio(cash={self.cash:.2f}, positions=[{pos_str}])"
