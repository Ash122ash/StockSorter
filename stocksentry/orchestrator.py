"""
StockSentry AI - Main Intelligence Orchestrator
Coordinates data retrieval, OHLC metrics, technical indicators, S/R zones,
news sentiment, 5-session forecast, crash-risk diagnostics, walk-forward validation,
and the Advanced Chart Analysis Suite (patterns, trendlines, divergences, scenario projections).
"""

from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from stocksentry.data_provider import default_provider
from stocksentry.ohlc_analyzer import compute_ohlc_metrics
from stocksentry.indicators import TechnicalIndicators
from stocksentry.support_resistance import sr_engine
from stocksentry.news_engine import news_engine
from stocksentry.forecasting import forecast_engine
from stocksentry.crash_risk import crash_engine
from stocksentry.validation import validator
from stocksentry.storage import storage

# Advanced Chart Analysis Modules
from stocksentry.chart.trendlines import trendline_engine
from stocksentry.chart.candlestick_patterns import candlestick_detector
from stocksentry.chart.chart_patterns import chart_pattern_engine
from stocksentry.chart.divergence import divergence_engine
from stocksentry.chart.forecast_overlay import generate_forecast_scenarios
from stocksentry.chart.signal_backtest import signal_backtester
from stocksentry.chart.analyst_reading import generate_analyst_reading


def run_full_analysis(symbol: str, period: str = "2y", force_refresh: bool = False) -> Dict[str, Any]:
    """
    Executes end-to-end quantitative, chart structural, and probabilistic analysis for a stock symbol.
    Returns rich data payload addressing all 11 key questions and advanced chart analysis features.
    """
    clean_sym = symbol.strip().upper()
    storage.log_audit(clean_sym, "RUN_FULL_ANALYSIS")

    # 1. Fetch Market Data
    df, meta = default_provider.fetch_history(clean_sym, period=period, force_refresh=force_refresh)
    if df is None or df.empty:
        raise ValueError(f"Could not retrieve price history for '{clean_sym}'. Verify ticker.")

    current_price = float(df["Close"].iloc[-1])

    # 2. OHLC & Return Performance Metrics (Q1, Q2, Q3)
    ohlc_data = compute_ohlc_metrics(df)

    # 3. Technical Indicators & Market Structure (Q4)
    df_augmented, tech_summary = TechnicalIndicators.calculate_all(df)

    # 4. Support and Resistance Zones (Q5, Q6)
    sr_data = sr_engine.analyze(df, current_price=current_price)

    # 5. News Intelligence & Sentiment (Q8)
    news_data = news_engine.fetch_and_analyze(clean_sym)
    agg_news_score = news_data.get("aggregate_score", 0.0)

    # 6. 5-Trading-Session Forecasting Engine (Q9, Q11)
    support_prices = [z["lower_boundary"] for z in sr_data.get("support_zones", [])[:2]]
    resist_prices = [z["upper_boundary"] for z in sr_data.get("resistance_zones", [])[:2]]

    forecast_data = forecast_engine.generate_forecast(
        df=df,
        symbol=clean_sym,
        news_score=agg_news_score,
        support_levels=support_prices,
        resistance_levels=resist_prices
    )

    # 7. Crash-Risk Engine (Q7)
    crash_data = crash_engine.assess_risk(
        df=df,
        symbol=clean_sym,
        support_zones=sr_data.get("support_zones", []),
        news_score=agg_news_score
    )

    # 8. Walk-Forward Forecast Validation (Q10)
    validation_data = validator.run_validation(df, symbol=clean_sym)

    # =========================================================================
    # ADVANCED CHART ANALYSIS MODULE SUITE
    # =========================================================================
    # 9. Geometric Trendlines & Channel Analysis
    trendline_data = trendline_engine.analyze(df)

    # 10. Classical Chart Patterns (Double Tops/Bottoms, Triangles, Flags, Wedges)
    detected_chart_patterns = chart_pattern_engine.detect_patterns(df)

    # 11. Single, Double & Triple Candlestick Patterns (Doji, Engulfing, Hammer, Morning Star, etc.)
    detected_candlestick_patterns = candlestick_detector.detect_all(df)

    # 12. RSI & MACD Divergence Analysis & MA Crossovers
    divergence_data = divergence_engine.analyze(df_augmented)

    # 13. Forward 5-Session Forecast Scenario Trajectories (P90, P50, P10)
    last_dt = df.index[-1]
    hv_pct = ohlc_data.get("historical_annualized_volatility_pct", 25.0)
    forecast_scenarios = generate_forecast_scenarios(
        last_date=last_dt,
        current_price=current_price,
        forecast_data=forecast_data,
        hv_20_pct=hv_pct
    )

    # 14. Chart Signal Chronological Out-of-Sample Backtesting
    signal_backtest_data = signal_backtester.backtest_all_signals(df_augmented)

    # 15. Technical Analyst's Reading Synthesis
    analyst_reading = generate_analyst_reading(
        symbol=clean_sym,
        current_price=current_price,
        currency=meta.get("currency", "INR"),
        ohlc=ohlc_data,
        tech=tech_summary,
        sr=sr_data,
        trendline_data=trendline_data,
        patterns=detected_chart_patterns,
        divergences=divergence_data,
        forecast=forecast_data
    )

    # 16. Synthesize Direct Answers to the 11 Core Questions
    q_and_a = {
        "q1_one_week_price_action": {
            "question": "What happened to its price over the last week (5 trading sessions)?",
            "answer": f"Price changed by {ohlc_data.get('five_session_return_pct', 'N/A')}% over the last 5 trading sessions. Latest close is {current_price:.2f}."
        },
        "q2_one_month_price_action": {
            "question": "What happened to its price over the last month (20 trading sessions)?",
            "answer": f"Price changed by {ohlc_data.get('twenty_session_return_pct', 'N/A')}% over the last 20 trading sessions with a cumulative return of {ohlc_data.get('cumulative_return_pct', 'N/A')}% over the analyzed horizon."
        },
        "q3_daily_open_and_close": {
            "question": "What were its daily opening and closing prices?",
            "answer": f"In the latest session ({meta.get('latest_session')}), Open was {ohlc_data.get('latest_open')} and Close was {ohlc_data.get('latest_close')} with a daily change of {ohlc_data.get('daily_return_pct')}% and an opening gap of {ohlc_data.get('opening_gap_pct')}%."
        },
        "q4_current_trend": {
            "question": "What is its current trend?",
            "answer": f"Technical indicator confluence classifies primary trend as '{tech_summary.get('primary_trend')}'. Market structure shows '{trendline_data.get('market_structure')}'."
        },
        "q5_historical_support_zones": {
            "question": "Which price zones have historically acted as support?",
            "answer": f"Detected {len(sr_data.get('support_zones', []))} support zones. Primary nearest support is at {sr_data.get('breakdown_risk_summary', {}).get('primary_support_zone', 'N/A')} with {sr_data.get('breakdown_risk_summary', {}).get('distance_to_primary_support_pct', 'N/A')}% downside buffer."
        },
        "q6_resistance_zones": {
            "question": "Which resistance zones may limit an upward move?",
            "answer": (
                f"Nearest confirmed resistance lies at "
                f"{sr_data.get('resistance_zones', [{}])[0].get('lower_boundary', 'N/A')} – "
                f"{sr_data.get('resistance_zones', [{}])[0].get('upper_boundary', 'N/A')} "
                f"({sr_data.get('resistance_zones', [{}])[0].get('distance_pct', 'N/A')}% above current price) "
                f"backed by {sr_data.get('resistance_zones', [{}])[0].get('meaningful_tests', 0)} historical tests."
                if sr_data.get("resistance_zones") else "No prominent resistance detected nearby."
            )
        },
        "q7_downside_scenarios_and_crash_risk": {
            "question": "How much could the stock fall under plausible downside scenarios?",
            "answer": f"Crash risk category is '{crash_data.get('risk_category')}' (Score: {crash_data.get('risk_score_100')}/100). Plausible 5-session downside stress scenario projects a decline of {crash_data.get('downside_stress_scenario', {}).get('percentage_fall')}% toward {crash_data.get('downside_stress_scenario', {}).get('projected_price')}."
        },
        "q8_recent_news_and_events": {
            "question": "What recent company-specific news could affect its price?",
            "answer": f"Retrieved {news_data.get('articles_count', 0)} news items with aggregate sentiment '{news_data.get('aggregate_sentiment')}' (Score: {news_data.get('aggregate_score')}). {len(news_data.get('high_impact_events', []))} high-impact events identified."
        },
        "q9_five_session_outlook": {
            "question": "Does available evidence favor a bullish, neutral or bearish outlook over next 5 trading sessions?",
            "answer": f"Model indicates '{forecast_data.get('outlook')}'. Probabilities: Bullish {forecast_data.get('probabilities', {}).get('bullish_pct', 0)}%, Neutral {forecast_data.get('probabilities', {}).get('neutral_pct', 0)}%, Bearish {forecast_data.get('probabilities', {}).get('bearish_pct', 0)}%."
        },
        "q10_system_reliability": {
            "question": "How reliable has the forecasting system been historically?",
            "answer": (
                f"Walk-forward out-of-sample directional accuracy is {validation_data.get('accuracy', {}).get('production_model_accuracy_pct', 'N/A')}% "
                f"(vs simple baseline {validation_data.get('accuracy', {}).get('simple_baseline_accuracy_pct', 'N/A')}%). "
                f"Bearish warning precision is {validation_data.get('bearish_alert_quality', {}).get('precision_pct', 'N/A')}%."
                if validation_data.get("status") != "INSUFFICIENT_HISTORY" else "Historical backtest requires minimum 120 sessions."
            )
        },
        "q11_invalidation_conditions": {
            "question": "What conditions would invalidate the current outlook?",
            "answer": "; ".join(forecast_data.get("invalidation_conditions", ["Price breakdown below key support or macro shock"]))
        }
    }

    # Chart data payloads (for Plotly and interactive charting)
    chart_df = df.tail(180).copy()
    chart_records = []
    for idx, row in chart_df.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        chart_records.append({
            "date": date_str,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
            "sma20": round(float(df_augmented.loc[idx, "SMA_20"]), 2) if "SMA_20" in df_augmented.columns and pd.notnull(df_augmented.loc[idx, "SMA_20"]) else None,
            "sma50": round(float(df_augmented.loc[idx, "SMA_50"]), 2) if "SMA_50" in df_augmented.columns and pd.notnull(df_augmented.loc[idx, "SMA_50"]) else None,
            "ema9": round(float(df_augmented.loc[idx, "EMA_9"]), 2) if "EMA_9" in df_augmented.columns and pd.notnull(df_augmented.loc[idx, "EMA_9"]) else None,
            "ema21": round(float(df_augmented.loc[idx, "EMA_21"]), 2) if "EMA_21" in df_augmented.columns and pd.notnull(df_augmented.loc[idx, "EMA_21"]) else None,
            "bb_upper": round(float(df_augmented.loc[idx, "BB_Upper"]), 2) if "BB_Upper" in df_augmented.columns and pd.notnull(df_augmented.loc[idx, "BB_Upper"]) else None,
            "bb_lower": round(float(df_augmented.loc[idx, "BB_Lower"]), 2) if "BB_Lower" in df_augmented.columns and pd.notnull(df_augmented.loc[idx, "BB_Lower"]) else None,
            "rsi14": round(float(df_augmented.loc[idx, "RSI_14"]), 2) if "RSI_14" in df_augmented.columns and pd.notnull(df_augmented.loc[idx, "RSI_14"]) else None,
            "macd_hist": round(float(df_augmented.loc[idx, "MACD_Hist"]), 2) if "MACD_Hist" in df_augmented.columns and pd.notnull(df_augmented.loc[idx, "MACD_Hist"]) else None
        })

    return {
        "symbol": clean_sym,
        "metadata": meta,
        "questions_and_answers": q_and_a,
        "ohlc_metrics": ohlc_data,
        "technical_analysis": tech_summary,
        "support_resistance": sr_data,
        "news_intelligence": news_data,
        "forecasting": forecast_data,
        "crash_risk": crash_data,
        "walk_forward_validation": validation_data,
        "trendline_analysis": trendline_data,
        "chart_patterns": detected_chart_patterns,
        "candlestick_patterns": detected_candlestick_patterns,
        "divergence_analysis": divergence_data,
        "forecast_scenarios": forecast_scenarios,
        "signal_backtesting": signal_backtest_data,
        "technical_analyst_reading": analyst_reading,
        "chart_data": chart_records
    }
