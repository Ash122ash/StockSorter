"""
StockSentry AI - System Configuration & Hyperparameters
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Storage
DB_PATH = os.getenv("STOCKSENTRY_DB_PATH", str(BASE_DIR / "data" / "stocksentry.db"))

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Technical Parameters
DEFAULT_PERIODS = {
    "5d": "5d",
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "max": "max"
}

# Support & Resistance Parameters
SR_CONFIG = {
    "swing_window": 5,           # Local window for swing high/low detection
    "cluster_tolerance_pct": 1.8, # Max % difference to cluster nearby levels
    "min_touches": 2,            # Minimum touches to qualify as a confirmed zone
    "zone_width_atr_mult": 0.5,   # ATR multiple for zone upper/lower buffer
    "recency_decay_halflife": 60, # Days for recency weight halflife
    "breakdown_threshold_pct": 1.0 # % below support to consider broken
}

# 5-Session Forecasting Thresholds (% return over next 5 trading days)
FORECAST_CONFIG = {
    "bullish_threshold_pct": 1.5,
    "bearish_threshold_pct": -1.5,
    "horizon_sessions": 5,
    "test_size_ratio": 0.25,
    "min_samples_required": 100
}

# Crash Risk Assessment Thresholds
CRASH_RISK_CONFIG = {
    "crash_threshold_pct": -5.0, # 5-session loss defining a severe downside/crash event
    "volatility_lookback": 20,
    "gap_down_threshold_pct": -1.5,
    "rvol_alert_level": 1.8,
    "atr_expansion_mult": 1.5
}
