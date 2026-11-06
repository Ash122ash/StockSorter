"""
StockSentry AI - Forecast Scenario Visualizer & Projection Engine
Generates forward 5-trading-session trajectory projections (Bullish, Base/Neutral, Bearish)
using model quantile estimations and historical volatility, plotted into future calendar dates.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_forecast_scenarios(
    last_date: datetime,
    current_price: float,
    forecast_data: Dict[str, Any],
    hv_20_pct: float
) -> Dict[str, Any]:
    """
    Constructs forward 5-trading-day date schedule and calculates trajectory lines for
    Bullish (P90), Base/Neutral (P50), and Bearish (P10) scenarios with confidence cones.
    """
    scenarios = forecast_data.get("scenarios", {})
    p10_pct = scenarios.get("downside_scenario_p10_pct", -2.5)
    p50_pct = scenarios.get("expected_median_p50_pct", 0.0)
    p90_pct = scenarios.get("upside_scenario_p90_pct", 2.5)

    # 1. Compute 5 future trading day dates (skipping weekends)
    future_dates: List[str] = []
    curr = last_date
    while len(future_dates) < 5:
        curr = curr + timedelta(days=1)
        # 5 = Saturday, 6 = Sunday
        if curr.weekday() < 5:
            future_dates.append(curr.strftime("%Y-%m-%d"))

    # Trajectory projection points over the 5 sessions (smooth nonlinear progression)
    # Day 1 to 5 fraction weights (t/5)^0.75
    weights = [(i / 5.0) ** 0.8 for i in range(1, 6)]

    bull_path = [round(current_price * (1.0 + (p90_pct / 100.0) * w), 2) for w in weights]
    base_path = [round(current_price * (1.0 + (p50_pct / 100.0) * w), 2) for w in weights]
    bear_path = [round(current_price * (1.0 + (p10_pct / 100.0) * w), 2) for w in weights]

    # Upper and lower uncertainty bounds (volatility envelope)
    daily_vol = (hv_20_pct / 100.0) / np.sqrt(252)
    upper_band = [round(bull_path[idx] * (1.0 + daily_vol * np.sqrt(idx + 1)), 2) for idx in range(5)]
    lower_band = [round(bear_path[idx] * (1.0 - daily_vol * np.sqrt(idx + 1)), 2) for idx in range(5)]

    last_date_str = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)

    return {
        "anchor_date": last_date_str,
        "anchor_price": round(current_price, 2),
        "future_dates": future_dates,
        "bullish_scenario": {
            "name": "Bullish Expansion (P90)",
            "target_return_pct": p90_pct,
            "target_price": bull_path[-1],
            "trajectory": bull_path,
            "color": "#10b981",
            "strengthen_condition": "Daily close above 21 EMA with expanding RVOL (>1.3x)",
            "invalidation_condition": "Failure to hold above opening baseline on day 1"
        },
        "base_scenario": {
            "name": "Base Equilibrium (P50 Median)",
            "target_return_pct": p50_pct,
            "target_price": base_path[-1],
            "trajectory": base_path,
            "color": "#f59e0b",
            "strengthen_condition": "Price oscillation inside recent Bollinger Band range",
            "invalidation_condition": "Directional breakout with expanding volume"
        },
        "bearish_scenario": {
            "name": "Bearish Downside Stress (P10)",
            "target_return_pct": p10_pct,
            "target_price": bear_path[-1],
            "trajectory": bear_path,
            "color": "#ef4444",
            "strengthen_condition": "Breach of nearest support band on high volume",
            "invalidation_condition": "Reclaim of 9 EMA with bullish divergence"
        },
        "uncertainty_cone": {
            "upper_band": upper_band,
            "lower_band": lower_band
        },
        "disclaimer": "Forecast paths represent probabilistic scenario envelopes, not deterministic prophecies. Plotted strictly into future sessions with dashed lines."
    }
