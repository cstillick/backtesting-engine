"""
Performance metrics for backtesting results.

Standard metrics used in quantitative finance:
- CAGR: Compound Annual Growth Rate
- Sharpe Ratio: Risk-adjusted return (annualized)
- Sortino Ratio: Like Sharpe but only penalizes downside volatility
- Max Drawdown: Largest peak-to-trough decline
- Win Rate: % of trades that were profitable
- Profit Factor: Gross profit / gross loss
- Calmar Ratio: CAGR / Max Drawdown

Usage:
  from src.metrics import compute_metrics
  metrics = compute_metrics(equity_series, trades_df, initial_cash=100_000)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def compute_cagr(equity: pd.Series, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound Annual Growth Rate."""
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    n_years = len(equity) / trading_days_per_year
    return float(total_return ** (1 / n_years) - 1)


def compute_sharpe(equity: pd.Series, risk_free_rate: float = 0.0, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualized Sharpe ratio.
    Uses daily returns. Risk-free rate default = 0 (simplification for demos).
    """
    daily_returns = equity.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    excess = daily_returns - (risk_free_rate / trading_days_per_year)
    return float(excess.mean() / excess.std() * np.sqrt(trading_days_per_year))


def compute_sortino(equity: pd.Series, risk_free_rate: float = 0.0, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualized Sortino ratio.
    Like Sharpe but uses downside deviation instead of total std.
    Better for strategies that have positive skew (occasional large gains).
    """
    daily_returns = equity.pct_change().dropna()
    excess = daily_returns - (risk_free_rate / trading_days_per_year)
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(trading_days_per_year))


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a fraction (negative number).
    e.g. -0.25 means the portfolio dropped 25% from its peak at some point.
    """
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Return the drawdown series (for plotting)."""
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def compute_calmar(equity: pd.Series) -> float:
    """CAGR / abs(Max Drawdown). Higher is better."""
    mdd = compute_max_drawdown(equity)
    if mdd == 0:
        return 0.0
    cagr = compute_cagr(equity)
    return float(cagr / abs(mdd))


def compute_win_rate(trades: pd.DataFrame) -> float:
    """
    Fraction of round-trip trades that were profitable.
    A round-trip = one BUY paired with one SELL.
    """
    if trades.empty:
        return 0.0

    buys = trades[trades["side"] == "BUY"].reset_index(drop=True)
    sells = trades[trades["side"] == "SELL"].reset_index(drop=True)

    n_pairs = min(len(buys), len(sells))
    if n_pairs == 0:
        return 0.0

    profits = []
    for i in range(n_pairs):
        buy_value = buys.loc[i, "value"] + buys.loc[i, "commission"]
        sell_value = sells.loc[i, "value"] - sells.loc[i, "commission"]
        profits.append(sell_value - buy_value)

    wins = sum(1 for p in profits if p > 0)
    return float(wins / n_pairs)


def compute_profit_factor(trades: pd.DataFrame) -> float:
    """Gross profit / gross loss. > 1 means strategy is profitable overall."""
    if trades.empty:
        return 0.0

    buys = trades[trades["side"] == "BUY"].reset_index(drop=True)
    sells = trades[trades["side"] == "SELL"].reset_index(drop=True)

    n_pairs = min(len(buys), len(sells))
    if n_pairs == 0:
        return 0.0

    gross_profit = 0.0
    gross_loss = 0.0
    for i in range(n_pairs):
        buy_value = buys.loc[i, "value"] + buys.loc[i, "commission"]
        sell_value = sells.loc[i, "value"] - sells.loc[i, "commission"]
        pnl = sell_value - buy_value
        if pnl > 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)

    return float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, initial_cash: float) -> dict:
    """
    Compute all performance metrics and return as a dict.

    Args:
        equity: Portfolio equity curve (pd.Series with DatetimeIndex)
        trades: Trade log DataFrame from Portfolio.get_trade_log_df()
        initial_cash: Starting capital

    Returns:
        Dict of metric name → value
    """
    total_return = (equity.iloc[-1] - initial_cash) / initial_cash

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(compute_cagr(equity) * 100, 2),
        "sharpe_ratio": round(compute_sharpe(equity), 3),
        "sortino_ratio": round(compute_sortino(equity), 3),
        "calmar_ratio": round(compute_calmar(equity), 3),
        "max_drawdown_pct": round(compute_max_drawdown(equity) * 100, 2),
        "win_rate_pct": round(compute_win_rate(trades) * 100, 1),
        "profit_factor": round(compute_profit_factor(trades), 3),
        "n_trades": len(trades[trades["side"] == "BUY"]) if not trades.empty else 0,
        "final_equity": round(equity.iloc[-1], 2),
        "initial_cash": round(initial_cash, 2),
    }
