"""
Unit & Integration Tests for StockSentry AI Platform
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from stocksentry.ohlc_analyzer import compute_ohlc_metrics
from stocksentry.indicators import TechnicalIndicators
from stocksentry.support_resistance import SupportResistanceEngine
from stocksentry.news_engine import NewsIntelligenceEngine
from stocksentry.forecasting import ForecastingEngine
from stocksentry.crash_risk import CrashRiskEngine
from stocksentry.validation import WalkForwardValidator


@pytest.fixture
def sample_ohlcv_df():
    """Generates synthetic 150-day trending OHLCV data for reproducible testing."""
    np.random.seed(42)
    n = 150
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    
    # Generate random walk with slight upward drift
    returns = np.random.normal(0.001, 0.015, n)
    price = 1000.0 * np.exp(np.cumsum(returns))
    
    high = price * (1.0 + np.random.uniform(0.002, 0.02, n))
    low = price * (1.0 - np.random.uniform(0.002, 0.02, n))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.randint(100000, 2000000, n)
    
    df = pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": price,
        "Volume": volume
    }, index=pd.to_datetime(dates))
    
    return df


def test_ohlc_metrics(sample_ohlcv_df):
    metrics = compute_ohlc_metrics(sample_ohlcv_df)
    assert "latest_close" in metrics
    assert "daily_return_pct" in metrics
    assert "five_session_return_pct" in metrics
    assert "twenty_session_return_pct" in metrics
    assert "maximum_drawdown_pct" in metrics
    assert metrics["latest_close"] > 0
    assert len(metrics["recent_ohlc_table"]) > 0


def test_technical_indicators(sample_ohlcv_df):
    augmented_df, summary = TechnicalIndicators.calculate_all(sample_ohlcv_df)
    assert "RSI_14" in augmented_df.columns
    assert "MACD_Line" in augmented_df.columns
    assert "BB_Upper" in augmented_df.columns
    assert "SMA_20" in augmented_df.columns
    assert "EMA_9" in augmented_df.columns
    assert summary["primary_trend"] in [
        "Strong Bullish", "Mildly Bullish / Neutral", 
        "Strong Bearish", "Mildly Bearish / Neutral", "Neutral / Sideways"
    ]
    assert 0 <= summary["indicators"]["RSI_14"] <= 100


def test_support_resistance_engine(sample_ohlcv_df):
    engine = SupportResistanceEngine()
    result = engine.analyze(sample_ohlcv_df)
    assert "support_zones" in result
    assert "resistance_zones" in result
    assert "breakdown_risk_summary" in result
    
    for sz in result["support_zones"]:
        assert sz["lower_boundary"] <= sz["upper_boundary"]
        assert sz["meaningful_tests"] >= 1
        assert "confidence_label" in sz


def test_news_sentiment_analysis():
    engine = NewsIntelligenceEngine()
    processed = engine._process_articles("INFY.NS", [
        {
            "headline": "Infosys beats Q3 net profit estimates by 8%, bags record mega deal",
            "publisher": "Reuters",
            "pub_timestamp": int(datetime.now().timestamp()),
            "publication_time": "2024-01-10",
            "link": "https://example.com"
        },
        {
            "headline": "SEBI issues probe notice regarding disclosure irregularities",
            "publisher": "Bloomberg",
            "pub_timestamp": int(datetime.now().timestamp()) - 3600,
            "publication_time": "2024-01-10",
            "link": "https://example.com"
        }
    ])
    assert len(processed) == 2
    assert processed[0]["sentiment_label"] == "Positive"
    assert processed[1]["sentiment_label"] == "Negative"


def test_forecasting_pipeline(sample_ohlcv_df):
    engine = ForecastingEngine()
    forecast = engine.generate_forecast(sample_ohlcv_df, symbol="TEST.NS")
    assert forecast["status"] == "VALID"
    assert "probabilities" in forecast
    probs = forecast["probabilities"]
    assert 99.0 <= (probs["bullish_pct"] + probs["neutral_pct"] + probs["bearish_pct"]) <= 101.0
    assert "scenarios" in forecast
    assert forecast["scenarios"]["downside_scenario_price"] > 0


def test_crash_risk_assessment(sample_ohlcv_df):
    engine = CrashRiskEngine()
    risk = engine.assess_risk(sample_ohlcv_df, symbol="TEST.NS")
    assert risk["risk_category"] in ["LOW", "ELEVATED", "HIGH", "SEVERE"]
    assert 0 <= risk["risk_score_100"] <= 100
    assert "downside_stress_scenario" in risk


def test_walk_forward_validation(sample_ohlcv_df):
    validator = WalkForwardValidator()
    results = validator.run_validation(sample_ohlcv_df, symbol="TEST.NS")
    assert results["status"] != "INSUFFICIENT_HISTORY"
    assert "accuracy" in results
    assert "production_model_accuracy_pct" in results["accuracy"]
    assert "bearish_alert_quality" in results
