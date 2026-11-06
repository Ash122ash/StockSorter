"""
Unit & Integration Tests for Advanced Chart Analysis Suite
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from stocksentry.chart.swing_points import detect_swing_extrema
from stocksentry.chart.trendlines import trendline_engine
from stocksentry.chart.candlestick_patterns import candlestick_detector
from stocksentry.chart.chart_patterns import chart_pattern_engine
from stocksentry.chart.divergence import divergence_engine
from stocksentry.chart.forecast_overlay import generate_forecast_scenarios
from stocksentry.chart.signal_backtest import signal_backtester
from stocksentry.chart.analyst_reading import generate_analyst_reading


@pytest.fixture
def synthetic_chart_df():
    """Generates synthetic 120-day OHLCV data with known swings and indicators."""
    np.random.seed(42)
    n = 120
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    
    # Sine wave price to guarantee clear swing highs and lows
    x = np.linspace(0, 4 * np.pi, n)
    trend = np.linspace(100, 150, n)
    price = trend + 15 * np.sin(x)
    
    high = price + np.random.uniform(1.0, 3.0, n)
    low = price - np.random.uniform(1.0, 3.0, n)
    open_p = low + (high - low) * 0.5
    volume = np.random.randint(500000, 2000000, n)
    
    df = pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": price,
        "Volume": volume
    }, index=pd.to_datetime(dates))
    
    # Calculate minimal indicators for divergence testing
    df["RSI_14"] = 50.0 + 20.0 * np.sin(x)
    df["MACD_Hist"] = 2.0 * np.cos(x)
    df["EMA_9"] = df["Close"].ewm(span=9).mean()
    df["EMA_21"] = df["Close"].ewm(span=21).mean()
    df["SMA_20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["SMA_50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["Vol_SMA_20"] = df["Volume"].rolling(20, min_periods=1).mean()
    
    return df


def test_swing_extrema_detection(synthetic_chart_df):
    result = detect_swing_extrema(synthetic_chart_df, window=4)
    assert len(result["swing_highs"]) >= 2
    assert len(result["swing_lows"]) >= 2
    for sh in result["swing_highs"]:
        assert sh["type"] == "high"
        assert sh["price"] > 0


def test_trendline_and_channel_engine(synthetic_chart_df):
    analysis = trendline_engine.analyze(synthetic_chart_df)
    assert "current_trend" in analysis
    assert "market_structure" in analysis
    assert "trendlines" in analysis
    assert isinstance(analysis["trendlines"], list)


def test_candlestick_pattern_detection(synthetic_chart_df):
    patterns = candlestick_detector.detect_all(synthetic_chart_df)
    assert isinstance(patterns, list)
    if patterns:
        p = patterns[0]
        assert "pattern" in p
        assert "bias" in p
        assert "price" in p


def test_chart_patterns_recognition(synthetic_chart_df):
    detected = chart_pattern_engine.detect_patterns(synthetic_chart_df)
    assert isinstance(detected, list)


def test_divergence_engine(synthetic_chart_df):
    div_results = divergence_engine.analyze(synthetic_chart_df)
    assert "rsi_status" in div_results
    assert "momentum_status" in div_results
    assert "rsi_current" in div_results
    assert "ma_crossovers" in div_results


def test_forecast_scenario_projections():
    last_dt = datetime(2023, 6, 1)
    current_p = 2500.0
    forecast_data = {
        "scenarios": {
            "downside_scenario_p10_pct": -4.2,
            "expected_median_p50_pct": 0.8,
            "upside_scenario_p90_pct": 5.1
        }
    }
    scen = generate_forecast_scenarios(last_dt, current_p, forecast_data, hv_20_pct=22.0)
    assert len(scen["future_dates"]) == 5
    assert len(scen["bullish_scenario"]["trajectory"]) == 5
    assert len(scen["bearish_scenario"]["trajectory"]) == 5
    assert scen["bullish_scenario"]["trajectory"][-1] > current_p
    assert scen["bearish_scenario"]["trajectory"][-1] < current_p


def test_chart_signal_backtesting(synthetic_chart_df):
    res = signal_backtester.backtest_all_signals(synthetic_chart_df)
    assert res["status"] == "VALID"
    assert "signal_summaries" in res


def test_analyst_reading_synthesis(synthetic_chart_df):
    reading = generate_analyst_reading(
        symbol="TEST.NS",
        current_price=150.0,
        currency="INR",
        ohlc={"relative_volume": 1.4},
        tech={"primary_trend": "Bullish Uptrend", "indicators": {"SMA_50": 140.0, "EMA_21": 145.0}},
        sr={"support_zones": [{"lower_boundary": 142.0, "upper_boundary": 144.0, "distance_description": "4% cushion", "meaningful_tests": 3, "average_reaction_pct": 5.2, "confidence_label": "Strong"}]},
        trendline_data={"current_trend": "Bullish Uptrend", "market_structure": "Higher Highs & Higher Lows"},
        patterns=[{"pattern": "Bull Flag", "bias": "Bullish Continuation", "confirmation_condition": "Break above 152.0", "measured_move_target": 165.0}],
        divergences={"rsi_current": 58.4, "rsi_status": "Bullish Expansion", "momentum_status": "Expanding"},
        forecast={"invalidation_conditions": ["Break below 142.0"]}
    )
    assert "panel_title" in reading
    assert "current_trend_analysis" in reading
    assert "support_zone_analysis" in reading
    assert "structural_pattern_analysis" in reading
    assert "potential_bullish_trigger" in reading
    assert "potential_bearish_trigger" in reading
