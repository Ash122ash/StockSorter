"""
StockSentry AI - Trendline & Market Structure Engine
Fits geometric trendlines across swing points, detects ascending/descending channels,
identifies HH/HL & LH/LL structures, and validates breakouts with volume confirmation.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from stocksentry.chart.swing_points import detect_swing_extrema


class TrendlineEngine:
    """
    Quantitative trendline and channel detection.
    Requires minimum touch count and geometric consistency; flags confirmed vs potential breakouts.
    """

    def __init__(self, touch_tolerance_pct: float = 1.0, min_touches: int = 2):
        self.touch_tolerance = touch_tolerance_pct / 100.0
        self.min_touches = min_touches

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects trendlines, channels, market structure (HH/HL or LH/LL), and breakout events.
        """
        if df is None or len(df) < 20:
            return {
                "trendlines": [],
                "channel": None,
                "structure": "Insufficient Data",
                "breakouts": [],
                "current_trend": "Unknown"
            }

        extrema = detect_swing_extrema(df, window=4)
        swing_highs = extrema["swing_highs"]
        swing_lows = extrema["swing_lows"]

        # 1. Market Structure Evaluation (HH/HL vs LH/LL)
        structure, current_trend = self._evaluate_market_structure(swing_highs, swing_lows)

        # 2. Fit Ascending (Support) and Descending (Resistance) Trendlines
        trendlines = []
        support_tl = self._fit_best_trendline(swing_lows, df, is_support=True)
        if support_tl:
            trendlines.append(support_tl)

        resist_tl = self._fit_best_trendline(swing_highs, df, is_support=False)
        if resist_tl:
            trendlines.append(resist_tl)

        # 3. Channel Detection
        channel = self._detect_channel(support_tl, resist_tl, df)

        # 4. Breakout / Breakdown Detection with Volume Confirmation
        breakouts = self._detect_breakouts(trendlines, df)

        return {
            "current_trend": current_trend,
            "market_structure": structure,
            "trendlines": trendlines,
            "channel": channel,
            "breakouts": breakouts
        }

    def _evaluate_market_structure(
        self,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Classifies Higher Highs/Higher Lows vs Lower Highs/Lower Lows."""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "Consolidation / Indeterminate Structure", "Neutral / Sideways"

        h1, h2 = swing_highs[-2]["price"], swing_highs[-1]["price"]
        l1, l2 = swing_lows[-2]["price"], swing_lows[-1]["price"]

        if h2 > h1 and l2 > l1:
            return "Higher Highs & Higher Lows (Confirmed Uptrend Structure)", "Bullish Uptrend"
        elif h2 < h1 and l2 < l1:
            return "Lower Highs & Lower Lows (Confirmed Downtrend Structure)", "Bearish Downtrend"
        elif h2 > h1 and l2 < l1:
            return "Expanding Megaphone / High Volatility Structure", "Neutral Volatile"
        elif h2 < h1 and l2 > l1:
            return "Contracting Wedge / Symmetrical Coiling", "Neutral Consolidating"
        else:
            return "Horizontal Trading Range Structure", "Neutral / Range-Bound"

    def _fit_best_trendline(
        self,
        swings: List[Dict[str, Any]],
        df: pd.DataFrame,
        is_support: bool
    ) -> Optional[Dict[str, Any]]:
        """Fits trendline connecting pairs of swing points and tests intermediary touches."""
        if len(swings) < 2:
            return None

        best_line = None
        max_score = -1.0
        n_candles = len(df)
        closes = df["Close"].values
        dates = df.index

        # Evaluate candidate pairs from the most recent swings
        candidates = swings[-6:] if len(swings) >= 6 else swings
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                p1 = candidates[i]
                p2 = candidates[j]
                idx1, idx2 = p1["index"], p2["index"]
                if idx2 - idx1 < 5:
                    continue

                # Linear equation: y = m*x + c
                m = (p2["price"] - p1["price"]) / (idx2 - idx1)
                c = p1["price"] - m * idx1

                # Slope sanity filter: filter out near-vertical lines
                price_ref = closes[idx1]
                slope_pct = (m / price_ref) * 100.0
                if abs(slope_pct) > 2.5: # excessive slope (>2.5% per session)
                    continue

                # Test validity and touch count across all subsequent candles
                touches = 0
                violations = 0
                for k in range(idx1, n_candles):
                    line_val = m * k + c
                    candle_high = df["High"].iloc[k]
                    candle_low = df["Low"].iloc[k]

                    if is_support:
                        # Close below line is violation
                        if closes[k] < line_val * (1.0 - self.touch_tolerance):
                            violations += 1
                        # Touch if candle low reaches near line
                        if abs(candle_low - line_val) / line_val <= self.touch_tolerance:
                            touches += 1
                    else:
                        # Close above line is violation
                        if closes[k] > line_val * (1.0 + self.touch_tolerance):
                            violations += 1
                        if abs(candle_high - line_val) / line_val <= self.touch_tolerance:
                            touches += 1

                # Require minimal violations
                if violations > 2:
                    continue

                score = (touches * 2.0) - violations + (0.5 * (idx2 / n_candles))
                if score > max_score and touches >= self.min_touches:
                    max_score = score
                    start_val = float(m * idx1 + c)
                    end_val = float(m * (n_candles - 1) + c)
                    proj_val = float(m * (n_candles + 4) + c) # 5 days forward

                    start_date_str = dates[idx1].strftime("%Y-%m-%d") if hasattr(dates[idx1], "strftime") else str(dates[idx1])
                    end_date_str = dates[-1].strftime("%Y-%m-%d") if hasattr(dates[-1], "strftime") else str(dates[-1])

                    line_type = "Ascending Support Trendline" if is_support and m > 0 else (
                        "Descending Support Trendline" if is_support else (
                            "Descending Resistance Trendline" if m < 0 else "Ascending Resistance Trendline"
                        )
                    )

                    best_line = {
                        "type": line_type,
                        "is_support": is_support,
                        "slope": round(float(m), 4),
                        "slope_pct_per_bar": round(float(slope_pct), 3),
                        "start_index": int(idx1),
                        "start_date": start_date_str,
                        "start_price": round(start_val, 2),
                        "end_index": int(n_candles - 1),
                        "end_date": end_date_str,
                        "end_price": round(end_val, 2),
                        "projected_5d_price": round(proj_val, 2),
                        "touches": touches,
                        "violations": violations,
                        "confidence": "Strong" if touches >= 3 and violations == 0 else "Moderate"
                    }

        return best_line

    def _detect_channel(
        self,
        support_tl: Optional[Dict[str, Any]],
        resist_tl: Optional[Dict[str, Any]],
        df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Identifies parallel price channels."""
        if not support_tl or not resist_tl:
            return None

        # Check slope similarity (parallel within 35%)
        s_slope = support_tl["slope"]
        r_slope = resist_tl["slope"]
        if (s_slope > 0 and r_slope > 0) or (s_slope < 0 and r_slope < 0):
            slope_diff = abs(s_slope - r_slope) / max(abs(s_slope), abs(r_slope), 1e-4)
            if slope_diff < 0.45:
                channel_type = "Ascending Bull Channel" if s_slope > 0 else "Descending Bear Channel"
                current_price = float(df["Close"].iloc[-1])
                upper_b = resist_tl["end_price"]
                lower_b = support_tl["end_price"]
                channel_height = upper_b - lower_b
                pos_pct = round(((current_price - lower_b) / max(channel_height, 1e-4)) * 100.0, 1)

                return {
                    "type": channel_type,
                    "channel_height_pct": round((channel_height / lower_b) * 100.0, 2),
                    "price_position_in_channel_pct": max(0.0, min(100.0, pos_pct)),
                    "interpretation": (
                        f"Trading inside a {channel_type}. "
                        f"Price is located at {pos_pct}% of the channel width."
                    )
                }
        return None

    def _detect_breakouts(
        self,
        trendlines: List[Dict[str, Any]],
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Detects recent breakouts or breakdowns with volume confirmation."""
        breakouts = []
        if len(df) < 5 or not trendlines:
            return breakouts

        vol_avg = float(df["Volume"].rolling(20, min_periods=5).mean().iloc[-1])
        latest_c = float(df["Close"].iloc[-1])
        latest_vol = float(df["Volume"].iloc[-1])
        rvol = latest_vol / vol_avg if vol_avg > 0 else 1.0

        for tl in trendlines:
            target_price = tl["end_price"]
            if tl["is_support"]:
                # Breakdown check: price below support line
                if latest_c < target_price * 0.992:
                    is_confirmed = rvol >= 1.3
                    breakouts.append({
                        "type": "Trendline Breakdown",
                        "trendline_type": tl["type"],
                        "level": target_price,
                        "current_price": latest_c,
                        "volume_confirmation": is_confirmed,
                        "rvol": round(rvol, 2),
                        "status": "Confirmed Breakdown" if is_confirmed else "Potential / Unconfirmed Breakdown",
                        "interpretation": (
                            f"Price sliced below support line at {target_price:.2f}. "
                            + ("High volume (RVOL " + str(round(rvol, 2)) + "x) validates institutional selling."
                               if is_confirmed else "Low volume suggests possible false break or bear trap.")
                        )
                    })
            else:
                # Breakout check: price above resistance line
                if latest_c > target_price * 1.008:
                    is_confirmed = rvol >= 1.3
                    breakouts.append({
                        "type": "Trendline Breakout",
                        "trendline_type": tl["type"],
                        "level": target_price,
                        "current_price": latest_c,
                        "volume_confirmation": is_confirmed,
                        "rvol": round(rvol, 2),
                        "status": "Confirmed Breakout" if is_confirmed else "Potential / Unconfirmed Breakout",
                        "interpretation": (
                            f"Price broke above descending resistance line at {target_price:.2f}. "
                            + ("Surging volume (RVOL " + str(round(rvol, 2)) + "x) confirms institutional breakout."
                               if is_confirmed else "Light volume cautions against a false breakout (bull trap).")
                        )
                    })

        return breakouts


# Global trendline engine instance
trendline_engine = TrendlineEngine()
