"""
Unit tests for the backtesting engine core components.

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.portfolio import Portfolio, Fill
from src.execution import ExecutionEngine, Order
from src.metrics import (
    compute_cagr, compute_sharpe, compute_max_drawdown,
    compute_win_rate, compute_metrics
)


# --- Fixtures ---

def make_price_series(n=100, start_price=100.0, trend=0.001):
    """Generate synthetic OHLCV data with a slight upward trend."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = [start_price * (1 + trend) ** i for i in range(n)]
    df = pd.DataFrame({
        "Open": prices,
        "High": [p * 1.005 for p in prices],
        "Low":  [p * 0.995 for p in prices],
        "Close": prices,
        "Volume": [1_000_000] * n,
    }, index=dates)
    return df


@pytest.fixture
def portfolio():
    return Portfolio(initial_cash=100_000.0)


@pytest.fixture
def engine():
    return ExecutionEngine(slippage_bps=0, commission_per_trade=0, commission_pct=0)


@pytest.fixture
def price_df():
    return make_price_series()


# --- Portfolio tests ---

class TestPortfolio:
    def test_initial_state(self, portfolio):
        assert portfolio.cash == 100_000.0
        assert portfolio.positions == {}
        assert portfolio.trade_log == []

    def test_buy_updates_cash_and_position(self, portfolio):
        fill = Fill(pd.Timestamp("2020-01-01"), "AAPL", "BUY", 10, 100.0, 0.0, 1000.0)
        portfolio.apply_fill(fill)
        assert portfolio.cash == 99_000.0
        assert portfolio.get_position("AAPL") == 10

    def test_sell_updates_cash_and_position(self, portfolio):
        buy = Fill(pd.Timestamp("2020-01-01"), "AAPL", "BUY", 10, 100.0, 0.0, 1000.0)
        portfolio.apply_fill(buy)
        sell = Fill(pd.Timestamp("2020-01-02"), "AAPL", "SELL", 10, 110.0, 0.0, 1100.0)
        portfolio.apply_fill(sell)
        assert portfolio.cash == pytest.approx(100_100.0)
        assert portfolio.get_position("AAPL") == 0

    def test_insufficient_cash_raises(self, portfolio):
        fill = Fill(pd.Timestamp("2020-01-01"), "AAPL", "BUY", 10000, 100.0, 0.0, 1_000_000.0)
        with pytest.raises(ValueError, match="Insufficient cash"):
            portfolio.apply_fill(fill)

    def test_oversell_raises(self, portfolio):
        buy = Fill(pd.Timestamp("2020-01-01"), "AAPL", "BUY", 5, 100.0, 0.0, 500.0)
        portfolio.apply_fill(buy)
        sell = Fill(pd.Timestamp("2020-01-02"), "AAPL", "SELL", 10, 100.0, 0.0, 1000.0)
        with pytest.raises(ValueError, match="Cannot sell"):
            portfolio.apply_fill(sell)

    def test_mark_to_market(self, portfolio):
        buy = Fill(pd.Timestamp("2020-01-01"), "AAPL", "BUY", 10, 100.0, 0.0, 1000.0)
        portfolio.apply_fill(buy)
        equity = portfolio.mark_to_market(pd.Timestamp("2020-01-02"), {"AAPL": 110.0})
        assert equity == pytest.approx(99_000.0 + 110.0 * 10)


# --- Execution engine tests ---

class TestExecutionEngine:
    def test_buy_executes_correctly(self, portfolio, engine):
        order = Order("AAPL", "BUY", 10)
        prices = {"AAPL": 100.0}
        fill = engine.execute(order, pd.Timestamp("2020-01-01"), prices, portfolio)
        assert fill is not None
        assert fill.side == "BUY"
        assert fill.quantity == 10
        assert portfolio.get_position("AAPL") == 10

    def test_sell_without_position_returns_none(self, portfolio, engine):
        order = Order("AAPL", "SELL", 10)
        prices = {"AAPL": 100.0}
        fill = engine.execute(order, pd.Timestamp("2020-01-01"), prices, portfolio)
        assert fill is None

    def test_slippage_applied(self, portfolio):
        engine = ExecutionEngine(slippage_bps=100, commission_per_trade=0, commission_pct=0)
        order = Order("AAPL", "BUY", 1)
        fill = engine.execute(order, pd.Timestamp("2020-01-01"), {"AAPL": 100.0}, portfolio)
        assert fill.price == pytest.approx(101.0)  # 100 * (1 + 0.01)

    def test_sells_before_buys(self, portfolio, engine):
        # Buy 10 first
        buy_fill = Fill(pd.Timestamp("2020-01-01"), "AAPL", "BUY", 10, 100.0, 0.0, 1000.0)
        portfolio.apply_fill(buy_fill)

        orders = [
            Order("AAPL", "BUY", 5),
            Order("AAPL", "SELL", 10),
        ]
        fills = engine.execute_all(orders, pd.Timestamp("2020-01-02"), {"AAPL": 100.0}, portfolio)
        # Sell should happen first, then buy
        assert fills[0].side == "SELL"
        assert fills[1].side == "BUY"


# --- Metrics tests ---

class TestMetrics:
    def make_equity(self, returns):
        initial = 100_000.0
        values = [initial]
        for r in returns:
            values.append(values[-1] * (1 + r))
        dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
        return pd.Series(values, index=dates)

    def test_cagr_flat(self):
        equity = self.make_equity([0.0] * 252)
        assert compute_cagr(equity) == pytest.approx(0.0, abs=0.001)

    def test_cagr_positive(self):
        daily_return = (1.1 ** (1/252)) - 1
        equity = self.make_equity([daily_return] * 252)
        assert compute_cagr(equity) == pytest.approx(0.1, abs=0.005)

    def test_max_drawdown_no_drawdown(self):
        equity = self.make_equity([0.01] * 100)
        assert compute_max_drawdown(equity) == pytest.approx(0.0, abs=0.001)

    def test_max_drawdown_known_value(self):
        # Goes up 50%, then drops 40%
        equity = pd.Series([100, 150, 90.0])
        mdd = compute_max_drawdown(equity)
        assert mdd == pytest.approx(-0.40, abs=0.01)

    def test_win_rate_all_wins(self):
        trades = pd.DataFrame([
            {"side": "BUY",  "value": 1000.0, "commission": 0.0},
            {"side": "SELL", "value": 1100.0, "commission": 0.0},
        ])
        assert compute_win_rate(trades) == 1.0

    def test_win_rate_all_losses(self):
        trades = pd.DataFrame([
            {"side": "BUY",  "value": 1000.0, "commission": 0.0},
            {"side": "SELL", "value": 900.0,  "commission": 0.0},
        ])
        assert compute_win_rate(trades) == 0.0
