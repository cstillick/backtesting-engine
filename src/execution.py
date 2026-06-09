"""
Order execution engine with slippage and commission simulation.

Slippage model: fixed basis points applied in the direction of the trade.
  - BUY fills slightly above the bar's close price (you pay more)
  - SELL fills slightly below (you receive less)

Commission model: flat fee per trade or % of notional (configurable).

This keeps things realistic without over-engineering. For production you'd
model bid/ask spreads, market impact, and partial fills.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import pandas as pd

from src.portfolio import Portfolio, Fill


@dataclass
class Order:
    """A trading order signal from a strategy."""
    ticker: str
    side: Literal["BUY", "SELL"]
    quantity: int           # number of shares
    order_type: str = "MARKET"  # only MARKET supported in this MVP

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"Side must be BUY or SELL, got {self.side}")


class ExecutionEngine:
    """
    Simulates order execution with slippage and commission.

    Args:
        slippage_bps: Slippage in basis points (1 bps = 0.01%).
                      Applied per side: buy fills high, sell fills low.
        commission_per_trade: Flat USD commission per trade (e.g. $1.00).
        commission_pct: Percentage of notional (e.g. 0.001 = 0.1%).
                        Commission = max(commission_per_trade, notional * commission_pct).
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_per_trade: float = 1.0,
        commission_pct: float = 0.001,
    ):
        self.slippage_bps = slippage_bps
        self.commission_per_trade = commission_per_trade
        self.commission_pct = commission_pct

    def _apply_slippage(self, price: float, side: str) -> float:
        """Adjust fill price for slippage."""
        slippage_multiplier = self.slippage_bps / 10_000
        if side == "BUY":
            return price * (1 + slippage_multiplier)
        else:
            return price * (1 - slippage_multiplier)

    def _compute_commission(self, notional: float) -> float:
        """Flat fee or percentage of notional, whichever is larger."""
        return max(self.commission_per_trade, notional * self.commission_pct)

    def execute(
        self,
        order: Order,
        date: pd.Timestamp,
        prices: dict[str, float],
        portfolio: Portfolio,
    ) -> Fill | None:
        """
        Attempt to execute an order against current prices.

        Returns a Fill if successful, None if the order is rejected
        (e.g. insufficient cash, zero price).
        """
        raw_price = prices.get(order.ticker)
        if raw_price is None or raw_price <= 0:
            print(f"[Execution] WARNING: No valid price for {order.ticker} on {date.date()}, skipping order")
            return None

        fill_price = self._apply_slippage(raw_price, order.side)
        notional = fill_price * order.quantity
        commission = self._compute_commission(notional)

        # Pre-flight check for BUY orders
        if order.side == "BUY":
            total_cost = notional + commission
            if total_cost > portfolio.cash:
                # Reduce quantity to what we can afford
                affordable_qty = int((portfolio.cash - commission) / fill_price)
                if affordable_qty <= 0:
                    print(f"[Execution] Insufficient cash for {order.ticker} BUY, skipping")
                    return None
                order = Order(order.ticker, order.side, affordable_qty)
                notional = fill_price * affordable_qty
                commission = self._compute_commission(notional)

        # Pre-flight check for SELL orders
        if order.side == "SELL":
            held = portfolio.get_position(order.ticker)
            if held == 0:
                print(f"[Execution] No position in {order.ticker} to sell, skipping")
                return None
            # Sell at most what we hold
            actual_qty = min(order.quantity, held)
            order = Order(order.ticker, order.side, actual_qty)
            notional = fill_price * actual_qty
            commission = self._compute_commission(notional)

        fill = Fill(
            date=date,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            value=notional,
        )

        portfolio.apply_fill(fill)
        return fill

    def execute_all(
        self,
        orders: list[Order],
        date: pd.Timestamp,
        prices: dict[str, float],
        portfolio: Portfolio,
    ) -> list[Fill]:
        """Execute a list of orders. Sells are processed before buys to free up cash."""
        sells = [o for o in orders if o.side == "SELL"]
        buys = [o for o in orders if o.side == "BUY"]

        fills = []
        for order in sells + buys:
            fill = self.execute(order, date, prices, portfolio)
            if fill:
                fills.append(fill)
        return fills
