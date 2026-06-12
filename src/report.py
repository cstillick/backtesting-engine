"""
Performance tearsheet generator.

Produces a two-panel plot:
  1. Equity curve (strategy vs. buy-and-hold benchmark)
  2. Drawdown chart

Usage:
  from src.report import generate_tearsheet
  generate_tearsheet(equity, trades, price_data, initial_cash, strategy_name="SMA Crossover")
"""

from __future__ import annotations
import os
import re
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from src.metrics import compute_metrics, compute_drawdown_series

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def build_benchmark(price_data: pd.DataFrame, initial_cash: float) -> pd.Series:
    """
    Buy-and-hold benchmark: invest all capital at the first close price.
    Returns an equity series aligned to price_data's index.
    """
    first_price = price_data["Close"].iloc[0]
    shares = initial_cash / first_price
    return price_data["Close"] * shares


def generate_tearsheet(
    equity: pd.Series,
    trades: pd.DataFrame,
    price_data: pd.DataFrame,
    initial_cash: float,
    strategy_name: str = "Strategy",
    ticker: str = "",
    output_path: str | None = None,
) -> dict:
    """
    Generate a tearsheet PNG and return the performance metrics dict.

    Args:
        equity: Strategy equity curve
        trades: Trade log DataFrame
        price_data: OHLCV DataFrame (for benchmark)
        initial_cash: Starting capital
        strategy_name: Label for chart legend
        ticker: Ticker symbol (for chart title)
        output_path: Where to save the PNG. Defaults to plots/tearsheet.png

    Returns:
        metrics dict (same as compute_metrics output)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if output_path is None:
        safe_name = re.sub(r"[^a-z0-9]+", "_", strategy_name.lower()).strip("_")
        output_path = os.path.join(PLOTS_DIR, f"tearsheet_{safe_name}.png")

    # --- Align benchmark to strategy equity index ---
    benchmark = build_benchmark(price_data, initial_cash)
    benchmark = benchmark.reindex(equity.index, method="ffill")

    # --- Compute metrics ---
    metrics = compute_metrics(equity, trades, initial_cash)

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    title = f"{strategy_name}"
    if ticker:
        title += f" — {ticker}"
    fig.suptitle(title, color="white", fontsize=14, y=0.98)

    # Panel 1: Equity curve
    ax1 = axes[0]
    ax1.plot(equity.index, equity.values, color="#00d4aa", linewidth=1.5, label=strategy_name)
    ax1.plot(benchmark.index, benchmark.values, color="#888888", linewidth=1.0, linestyle="--", label="Buy & Hold")
    ax1.set_ylabel("Portfolio Value ($)", color="white", fontsize=10)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=9)
    ax1.grid(True, color="#2a2a2a", linewidth=0.5)

    # Annotate final metrics
    metric_text = (
        f"CAGR: {metrics['cagr_pct']:+.1f}%  |  "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
        f"Max DD: {metrics['max_drawdown_pct']:.1f}%  |  "
        f"Win Rate: {metrics['win_rate_pct']:.0f}%  |  "
        f"Trades: {metrics['n_trades']}"
    )
    ax1.set_title(metric_text, color="#aaaaaa", fontsize=9, pad=6)

    # Panel 2: Drawdown
    ax2 = axes[1]
    drawdown = compute_drawdown_series(equity)
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="#ff4444", alpha=0.6, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)", color="white", fontsize=10)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.set_xlabel("Date", color="white", fontsize=10)
    ax2.grid(True, color="#2a2a2a", linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Tearsheet saved to {output_path}")

    return metrics


def print_metrics_table(metrics: dict, strategy_name: str = ""):
    """Pretty-print metrics to console."""
    separator = "─" * 45
    print(f"\n{separator}")
    if strategy_name:
        print(f"  {strategy_name}")
    print(separator)
    rows = [
        ("Total Return",    f"{metrics['total_return_pct']:+.2f}%"),
        ("CAGR",            f"{metrics['cagr_pct']:+.2f}%"),
        ("Sharpe Ratio",    f"{metrics['sharpe_ratio']:.3f}"),
        ("Sortino Ratio",   f"{metrics['sortino_ratio']:.3f}"),
        ("Calmar Ratio",    f"{metrics['calmar_ratio']:.3f}"),
        ("Max Drawdown",    f"{metrics['max_drawdown_pct']:.2f}%"),
        ("Win Rate",        f"{metrics['win_rate_pct']:.1f}%"),
        ("Profit Factor",   f"{metrics['profit_factor']:.3f}"),
        ("# Trades",        str(metrics['n_trades'])),
        ("Final Equity",    f"${metrics['final_equity']:,.2f}"),
    ]
    for label, value in rows:
        print(f"  {label:<20} {value:>10}")
    print(separator)
