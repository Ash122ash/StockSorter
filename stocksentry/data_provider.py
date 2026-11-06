"""
StockSentry AI - Market Data Provider Interface & Implementations
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
from stocksentry.storage import storage

logger = logging.getLogger(__name__)


class MarketDataError(Exception):
    """Raised when market data cannot be retrieved or validated."""
    pass


class BaseMarketDataProvider(ABC):
    """Abstract interface allowing replacement of yfinance with a licensed institutional API."""

    @abstractmethod
    def fetch_history(self, symbol: str, period: str = "2y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch OHLCV historical dataframe and metadata.
        Returns (DataFrame, metadata_dict).
        """
        pass

    @abstractmethod
    def validate_ticker(self, symbol: str) -> bool:
        """Validate whether the ticker exists on the exchange."""
        pass


class YFinanceDataProvider(BaseMarketDataProvider):
    """
    Production-grade yfinance market data service with:
    - Retries and exponential backoff
    - Chronological sorting and deduplication
    - Missing data filtering
    - Stale data alerts
    - Non-real-time disclaimer
    - Multi-period caching
    """

    def __init__(self, max_retries: int = 3, timeout_seconds: int = 15):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def clean_symbol(self, symbol: str) -> str:
        s = symbol.strip().upper()
        # If user typed standard Indian stock without suffix, don't force if already has dot
        return s

    def validate_ticker(self, symbol: str) -> bool:
        symbol = self.clean_symbol(symbol)
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info and getattr(fast_info, "last_price", None) is not None:
                return True
            hist = ticker.history(period="5d")
            return not hist.empty
        except Exception:
            return False

    def fetch_history(self, symbol: str, period: str = "2y", force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        clean_sym = self.clean_symbol(symbol)

        # 1. Check cache first unless forced
        if not force_refresh:
            cached = storage.get_ohlc_cache(clean_sym, period)
            if cached and "records" in cached and "metadata" in cached:
                df = pd.DataFrame(cached["records"])
                if not df.empty and "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    df.sort_index(inplace=True)
                    meta = cached["metadata"]
                    meta["is_cached"] = True
                    return df, meta

        # 2. Fetch from yfinance with retry mechanism
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                ticker = yf.Ticker(clean_sym)
                df = ticker.history(period=period, auto_adjust=False)
                if df is None or df.empty:
                    # Try alternative format if symbol has no suffix
                    if "." not in clean_sym:
                        alt_sym = f"{clean_sym}.NS"
                        ticker = yf.Ticker(alt_sym)
                        df = ticker.history(period=period, auto_adjust=False)
                        if not df.empty:
                            clean_sym = alt_sym

                if df is None or df.empty:
                    raise MarketDataError(f"No price history found for symbol '{clean_sym}'. Check ticker or exchange suffix (e.g., RELIANCE.NS, AAPL).")

                # Clean and structure DataFrame
                df = df.copy()
                df.reset_index(inplace=True)
                
                # Standardize Date column
                date_col = "Date" if "Date" in df.columns else df.columns[0]
                df["Date"] = pd.to_datetime(df[date_col])
                # Remove timezone for clean tabular consistency
                if hasattr(df["Date"].dt, "tz") and df["Date"].dt.tz is not None:
                    df["Date"] = df["Date"].dt.tz_localize(None)

                df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
                df.sort_values("Date", ascending=True, inplace=True)
                df.set_index("Date", inplace=True)

                # Ensure required OHLCV columns exist
                for req in ["Open", "High", "Low", "Close", "Volume"]:
                    if req not in df.columns:
                        raise MarketDataError(f"Missing required price column '{req}' from market data feed.")

                # Drop invalid rows with NaN or zero Close
                df = df[df["Close"].notnull() & (df["Close"] > 0)].copy()

                if "Adj Close" not in df.columns:
                    df["Adj Close"] = df["Close"]

                # Check for corporate action splits/dividends
                splits_detected = False
                if "Stock Splits" in df.columns and (df["Stock Splits"] > 0).any():
                    splits_detected = True

                # Metadata construction
                last_session = df.index[-1].strftime("%Y-%m-%d")
                latest_close = float(df["Close"].iloc[-1])
                
                # Check for stale data (e.g. last session > 5 calendar days ago)
                days_since_last = (datetime.now() - df.index[-1]).days
                is_stale = days_since_last > 4

                # Exchange and currency detection
                currency = "INR" if clean_sym.endswith((".NS", ".BO")) else "USD"
                
                metadata = {
                    "symbol": clean_sym,
                    "period_requested": period,
                    "total_sessions": len(df),
                    "first_session": df.index[0].strftime("%Y-%m-%d"),
                    "latest_session": last_session,
                    "latest_close": round(latest_close, 2),
                    "currency": currency,
                    "is_cached": False,
                    "is_stale": is_stale,
                    "days_since_last_session": days_since_last,
                    "corporate_splits_detected": splits_detected,
                    "data_disclaimer": "Prices are delayed or historical close records (not real-time). Never interpreted as live tick feed.",
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                }

                # Save to cache
                records_to_cache = df.reset_index().copy()
                records_to_cache["Date"] = records_to_cache["Date"].dt.strftime("%Y-%m-%d")
                storage.save_ohlc_cache(clean_sym, period, {
                    "records": records_to_cache.to_dict(orient="records"),
                    "metadata": metadata
                })

                return df, metadata

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed for {clean_sym}: {e}")
                time.sleep(1.0 * attempt)

        raise MarketDataError(f"Market data request failed for '{clean_sym}' after {self.max_retries} attempts: {last_error}")


# Global default provider
default_provider = YFinanceDataProvider()
