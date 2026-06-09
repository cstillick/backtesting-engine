"""
Demo script — run both strategies and generate tearsheets.

Usage:
  python run_backtest.py
  python run_backtest.py --ticker SPY --start 2018-01-01 --end 2024-01-01
"""

import argparse
from src.data_loader import DataLoader
from src.engine import Backtest
from src.strategies.sma_crossover import SMACrossover
from src.strategies.rsi import RSIStrategy
from src.report import generate_tearsheet, print_metrics_table


def run(ticker: str, start: str, end: str, initial_cash: float = 100_000.0):
    loader = DataLoader()
    data = loader.get(ticker, start=start, end=end)

    strategies = [
        ("SMA Crossover (20/50)", SMACrossover(ticker, fast=20, slow=50)),
        ("RSI Mean Reversion",    RSIStrategy(ticker, period=14, oversold=30, overbought=70)),
    ]

    all_metrics = {}
    for name, strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")

        bt = Backtest(strategy=strategy, data=data, initial_cash=initial_cash)
        results = bt.run()

        metrics = generate_tearsheet(
            equity=results["equity"],
            trades=results["trades"],
            price_data=data,
            initial_cash=initial_cash,
            strategy_name=name,
            ticker=ticker,
        )
        print_metrics_table(metrics, strategy_name=name)
        all_metrics[name] = metrics

    # Summary comparison
    print(f"\n{'='*50}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*50}")
    print(f"  {'Strategy':<28} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>8}")
    print(f"  {'-'*52}")
    for name, m in all_metrics.items():
        print(f"  {name:<28} {m['cagr_pct']:>7.1f}% {m['sharpe_ratio']:>8.2f} {m['max_drawdown_pct']:>7.1f}%")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtests on a ticker")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--start",  default="2018-01-01", help="Start date (default: 2018-01-01)")
    parser.add_argument("--end",    default="2024-01-01", help="End date (default: 2024-01-01)")
    parser.add_argument("--cash",   default=100_000, type=float, help="Initial capital (default: 100000)")
    args = parser.parse_args()

    run(ticker=args.ticker, start=args.start, end=args.end, initial_cash=args.cash)
