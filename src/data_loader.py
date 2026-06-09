"""
Market data loader using yFinance with local CSV caching.

Fetches OHLCV (Open, High, Low, Close, Volume) data for any ticker and date range.
Caches to CSV so repeated runs don't hit the network.

Usage:
  from src.data_loader import DataLoader
  loader = DataLoader()
  df = loader.get("AAPL", start="2020-01-01", end="2024-01-01")
"""

import os
import hashlib
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


class DataLoader:
    """
    Loads historical OHLCV data from yFinance with CSV caching.
    Cache key = ticker + start + end, so stale data is never silently returned.
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, ticker: str, start: str, end: str) -> str:
        key = f"{ticker}_{start}_{end}"
        filename = hashlib.md5(key.encode()).hexdigest()[:12] + f"_{ticker}.csv"
        return os.path.join(self.cache_dir, filename)

    def get(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame for ticker between start and end dates.

        Args:
            ticker: e.g. "AAPL", "SPY", "BTC-USD"
            start: ISO date string, e.g. "2020-01-01"
            end: ISO date string, e.g. "2024-01-01"
            interval: yFinance interval ("1d", "1wk", "1mo")
            force_refresh: Bypass cache and re-download

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index: DatetimeIndex (timezone-naive)
        """
        cache_path = self._cache_path(ticker, start, end)

        if not force_refresh and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"[DataLoader] Loaded {ticker} from cache ({len(df)} rows)")
            return df

        print(f"[DataLoader] Downloading {ticker} ({start} → {end})...")
        raw = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)

        if raw.empty:
            raise ValueError(f"No data returned for {ticker} between {start} and {end}")

        # Flatten multi-level columns if present (yfinance quirk with single tickers)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Normalize to standard column names
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)  # timezone-naive
        df.index.name = "Date"

        df.to_csv(cache_path)
        print(f"[DataLoader] Saved to cache ({len(df)} rows)")
        return df

    def get_multiple(self, tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers, returning a dict keyed by ticker."""
        return {ticker: self.get(ticker, start, end) for ticker in tickers}
