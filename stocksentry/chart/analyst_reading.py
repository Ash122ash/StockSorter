"""
StockSentry AI - Technical Analyst's Reading Synthesis Engine
Generates human-readable, institutional-grade analytical commentary derived
strictly from calculated price action facts, algorithmic detections, and quantitative indicators.
"""

from typing import Dict, Any, List


def generate_analyst_reading(
    symbol: str,
    current_price: float,
    currency: str,
    ohlc: Dict[str, Any],
    tech: Dict[str, Any],
    sr: Dict[str, Any],
    trendline_data: Dict[str, Any],
    patterns: List[Dict[str, Any]],
    divergences: Dict[str, Any],
    forecast: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesizes complete 'Technical Analyst's Reading' report strictly from computed data.
    """
    curr = "₹" if currency == "INR" else "$"
    
    # 1. Current Trend & Structure
    trend = trendline_data.get("current_trend", tech.get("primary_trend", "Neutral"))
    structure = trendline_data.get("market_structure", tech.get("market_structure", "Range-Bound"))

    trend_narrative = (
        f"{symbol} is currently in a confirmed {trend}. Market structure exhibits {structure}. "
        f"Price is trading {'above' if current_price > tech.get('indicators', {}).get('SMA_50', current_price) else 'below'} "
        f"its 50-day moving average ({curr}{tech.get('indicators', {}).get('SMA_50', 0):.2f}) and "
        f"{'above' if current_price > tech.get('indicators', {}).get('EMA_21', current_price) else 'below'} its 21-day EMA."
    )

    # 2. Nearest Support & Resistance
    s_zones = sr.get("support_zones", [])
    r_zones = sr.get("resistance_zones", [])

    if s_zones:
        sz = s_zones[0]
        support_narrative = (
            f"Nearest high-conviction support zone sits at {curr}{sz['lower_boundary']} – {curr}{sz['upper_boundary']} "
            f"({sz['distance_description']}), backed by {sz['meaningful_tests']} verified tests with an average rebound "
            f"of +{sz['average_reaction_pct']}%. Classified as '{sz['confidence_label']}'."
        )
    else:
        support_narrative = "No major support zone identified within immediate price vicinity."

    if r_zones:
        rz = r_zones[0]
        resist_narrative = (
            f"Immediate overhead supply ceiling is established at {curr}{rz['lower_boundary']} – {curr}{rz['upper_boundary']} "
            f"({rz['distance_description']}), validated by {rz['meaningful_tests']} prior rejections. "
            f"Classified as '{rz['confidence_label']}'."
        )
    else:
        resist_narrative = "Trading in blue-sky territory with no overhead horizontal resistance."

    # 3. Most Important Pattern
    if patterns:
        p = patterns[0]
        pattern_narrative = (
            f"Algorithmic structure detected: **{p['pattern']}** ({p['bias']}). "
            f"Confirmation condition: {p['confirmation_condition']} "
            + (f"Measured-move price objective: {curr}{p['measured_move_target']}." if "measured_move_target" in p else "")
        )
    else:
        pattern_narrative = "No multi-week classical geometry currently active; price is respecting horizontal channels."

    # 4. RSI & MACD Momentum Interpretation
    rsi_val = divergences.get("rsi_current", 50.0)
    rsi_status = divergences.get("rsi_status", "Neutral")
    momentum_status = divergences.get("momentum_status", "Neutral")

    rsi_div_list = divergences.get("rsi_divergences", [])
    div_note = ""
    if rsi_div_list:
        div_note = f" Critical alert: {rsi_div_list[0]['significance']}."

    momentum_narrative = (
        f"RSI (14) stands at {rsi_val:.1f} ({rsi_status}). MACD histogram reflects {momentum_status}.{div_note}"
    )

    # 5. Volume Confirmation
    rvol = ohlc.get("relative_volume", 1.0)
    if rvol >= 1.5:
        volume_narrative = f"Exceptional institutional volume participation (RVOL {rvol:.2f}x of 20-day mean), indicating active accumulation/distribution."
    elif rvol <= 0.7:
        volume_narrative = f"Subdued trading volume (RVOL {rvol:.2f}x), cautioning that recent moves lack institutional conviction."
    else:
        volume_narrative = f"Normal trading liquidity with RVOL at {rvol:.2f}x of 20-day baseline."

    # 6. Bullish & Bearish Triggers
    if r_zones:
        bull_trigger = f"A daily close above overhead resistance at {curr}{r_zones[0]['upper_boundary']} accompanied by expanding volume (>1.3x RVOL)."
    else:
        bull_trigger = f"A sustained hold above 9-day EMA with expanding MACD positive histogram."

    if s_zones:
        bear_trigger = f"A confirmed session close below the primary support floor at {curr}{s_zones[0]['lower_boundary']}."
    else:
        bear_trigger = f"A high-volume breach below the 50-day moving average."

    # 7. Invalidation Conditions
    invalidation_conditions = forecast.get("invalidation_conditions", [
        f"A break below {curr}{s_zones[0]['lower_boundary'] if s_zones else 'recent swing low'}"
    ])

    return {
        "panel_title": "Technical Analyst's Reading",
        "current_trend_analysis": trend_narrative,
        "support_zone_analysis": support_narrative,
        "resistance_zone_analysis": resist_narrative,
        "structural_pattern_analysis": pattern_narrative,
        "momentum_and_divergence_analysis": momentum_narrative,
        "volume_confirmation_analysis": volume_narrative,
        "potential_bullish_trigger": bull_trigger,
        "potential_bearish_trigger": bear_trigger,
        "forecast_invalidation_conditions": invalidation_conditions,
        "methodology_declaration": (
            "This reading is synthesized deterministically from algorithmic market structure detections, "
            "empirical price action formulas, and statistical momentum indicators. No subjective or generative hallucination."
        )
    }
