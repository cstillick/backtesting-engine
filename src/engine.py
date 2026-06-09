"""
Backtesting engine — the main event loop.

Iterates bar-by-bar through historical price data, calling the strategy's
on_bar() at each step and routing orders through the execution engine.

Usage:
  from src.engine import Backtest
  from src.strategies.sma_crossover import SMACrossover

  bt = Backtest(
      strategy=SMACrossover("AAPL", fast=20, slow=50),
      data=price_df,
      initial_cash=100_000,
  )
  results = bt.run()
"""

from __future__ import annotations
import pandas as pd

from src.portfolio import Portfolio
from src.execution import ExecutionEngine
from src.strategy import BaseStrategy


class Backtest:
    """
    Drives the bar-by-bar simulation.

    Args:
        strategy: A BaseStrategy subclass instance
        data: OHLCV DataFrame with DatetimeIndex
        initial_cash: Starting capital in USD (default $100,000)
        slippage_bps: Slippage in basis points per trade (default 5 bps)
        commission_per_trade: Flat commission in USD (default $1.00)
        commission_pct: Commission as % of notional (default 0.1%)
        warmup_bars: Number of bars to skip at the start (for indicator warmup).
                     The strategy runs from bar 0, but equity curve starts here.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
        slippage_bps: float = 5.0,
        commission_per_trade: float = 1.0,
        commission_pct: float = 0.001,
    ):
        self.strategy = strategy
        self.data = data.sort_index()
        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.engine = ExecutionEngine(
            slippage_bps=slippage_bps,
            commission_per_trade=commission_per_trade,
            commission_pct=commission_pct,
        )

    def run(self) -> dict:
        """
        Run the backtest. Returns a results dict with:
        - portfolio: Portfolio object (equity curve, trade log, etc.)
        - equity: pd.Series of equity over time
        - trades: pd.DataFrame of all fills
        """
        # Give strategy a chance to precompute indicators on full history
        self.strategy.on_start(self.data)

        dates = self.data.index

        for i, date in enumerate(dates):
            # Pass historical data up to and including today (no lookahead bias)
            window = self.data.iloc[: i + 1]

            # Get orders from strategy
            orders = self.strategy.on_bar(date, window, self.portfolio)

            # Execute orders at today's close price
            prices = {self.strategy.ticker: float(self.data.loc[date, "Close"])}
            if orders:
                self.engine.execute_all(orders, date, prices, self.portfolio)

            # Mark portfolio to market at close
            self.portfolio.mark_to_market(date, prices)

        self.strategy.on_end(self.portfolio)

        equity = self.portfolio.get_equity_series()
        trades = self.portfolio.get_trade_log_df()

        n_trades = len(trades)
        print(f"\n[Backtest] Complete — {len(dates)} bars, {n_trades} trades")
        print(f"[Backtest] Final equity: ${equity.iloc[-1]:,.2f} (started: ${self.portfolio.initial_cash:,.2f})")

        return {
            "portfolio": self.portfolio,
            "equity": equity,
            "trades": trades,
        }
