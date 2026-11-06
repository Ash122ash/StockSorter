"""
StockSentry AI - Candlestick Pattern Detection Engine
Identifies standard single, double, and triple candlestick reversal and continuation patterns
using geometric candle anatomy and preceding trend context.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class CandlestickPatternDetector:
    """
    Automated algorithmic candlestick anatomy analyzer.
    Evaluates body ratios, shadow proportions, and prior trend context.
    """

    def detect_all(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scans DataFrame and returns list of all detected candlestick patterns.
        """
        if df is None or len(df) < 5:
            return []

        patterns: List[Dict[str, Any]] = []
        d = df.copy()

        opens = d["Open"].values
        highs = d["High"].values
        lows = d["Low"].values
        closes = d["Close"].values
        volumes = d["Volume"].values if "Volume" in d.columns else np.ones(len(d))
        dates = d.index

        # Preceding 5-day trend helper
        def get_trend_before(idx: int) -> str:
            if idx < 5:
                return "Neutral"
            ret = (closes[idx-1] / closes[idx-5]) - 1.0
            if ret > 0.015:
                return "Uptrend"
            elif ret < -0.015:
                return "Downtrend"
            return "Neutral"

        n = len(d)
        for i in range(2, n):
            c_open, c_high, c_low, c_close = opens[i], highs[i], lows[i], closes[i]
            c_body = abs(c_close - c_open)
            c_range = max(c_high - c_low, 1e-4)
            c_upper_shadow = c_high - max(c_open, c_close)
            c_lower_shadow = min(c_open, c_close) - c_low
            is_bullish = c_close >= c_open

            prior_trend = get_trend_before(i)
            date_str = dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], "strftime") else str(dates[i])

            # 1. Doji (Body <= 10% of total range)
            if c_body / c_range <= 0.10:
                patterns.append({
                    "date": date_str,
                    "index": i,
                    "pattern": "Doji",
                    "bias": "Neutral / Indecision",
                    "price": round(float(c_close), 2),
                    "significance": "Equilibrium between buyers and sellers; warns of potential trend pause or reversal.",
                    "failure_risk": "High false-signal frequency during low-volume sessions."
                })

            # 2. Hammer (Small body near top, lower shadow >= 2x body, tiny upper shadow, in Downtrend)
            if prior_trend == "Downtrend" and c_lower_shadow >= 2.0 * c_body and c_upper_shadow <= 0.15 * c_range and c_body / c_range >= 0.15:
                patterns.append({
                    "date": date_str,
                    "index": i,
                    "pattern": "Hammer",
                    "bias": "Bullish Reversal",
                    "price": round(float(c_close), 2),
                    "significance": "Intraday seller capitulation followed by fierce institutional buying back to highs.",
                    "failure_risk": "Invalidated if subsequent session breaches the lower shadow low."
                })

            # 3. Inverted Hammer (Small body near bottom, upper shadow >= 2x body, tiny lower shadow, in Downtrend)
            if prior_trend == "Downtrend" and c_upper_shadow >= 2.0 * c_body and c_lower_shadow <= 0.15 * c_range and c_body / c_range >= 0.15:
                patterns.append({
                    "date": date_str,
                    "index": i,
                    "pattern": "Inverted Hammer",
                    "bias": "Bullish Reversal Alert",
                    "price": round(float(c_close), 2),
                    "significance": "Early buyer test of higher liquidity; demands bullish confirmation on next session.",
                    "failure_risk": "Can resolve into continuation if bears overwhelm the upper wick."
                })

            # 4. Shooting Star (Small body near bottom, upper shadow >= 2x body, tiny lower shadow, in Uptrend)
            if prior_trend == "Uptrend" and c_upper_shadow >= 2.0 * c_body and c_lower_shadow <= 0.15 * c_range and c_body / c_range >= 0.15:
                patterns.append({
                    "date": date_str,
                    "index": i,
                    "pattern": "Shooting Star",
                    "bias": "Bearish Reversal",
                    "price": round(float(c_close), 2),
                    "significance": "Bulls failed to hold intra-session highs; heavy overhead selling rejection.",
                    "failure_risk": "Sustained volume break above the wick invalidates the reversal."
                })

            # 5. Hanging Man (Small body near top, lower shadow >= 2x body, in Uptrend)
            if prior_trend == "Uptrend" and c_lower_shadow >= 2.0 * c_body and c_upper_shadow <= 0.15 * c_range and c_body / c_range >= 0.15:
                patterns.append({
                    "date": date_str,
                    "index": i,
                    "pattern": "Hanging Man",
                    "bias": "Bearish Reversal Alert",
                    "price": round(float(c_close), 2),
                    "significance": "Severe intraday liquidity vacuum signals exhaustion at top.",
                    "failure_risk": "Requires confirmation close below body on next trading session."
                })

            # Double Candle Patterns (i and i-1)
            prev_open, prev_close = opens[i-1], closes[i-1]
            prev_body = abs(prev_close - prev_open)
            prev_bullish = prev_close >= prev_open

            # 6. Bullish Engulfing (Prev candle red, current candle green engulfing prev body, in Downtrend)
            if prior_trend == "Downtrend" and not prev_bullish and is_bullish:
                if c_open <= prev_close and c_close >= prev_open and c_body > prev_body * 1.1:
                    patterns.append({
                        "date": date_str,
                        "index": i,
                        "pattern": "Bullish Engulfing",
                        "bias": "Strong Bullish Reversal",
                        "price": round(float(c_close), 2),
                        "significance": "Bulls decisively overwhelm prior session selling pressure with expansion range.",
                        "failure_risk": "Invalidated if price trades below the engulfing candle low."
                    })

            # 7. Bearish Engulfing (Prev candle green, current candle red engulfing prev body, in Uptrend)
            if prior_trend == "Uptrend" and prev_bullish and not is_bullish:
                if c_open >= prev_close and c_close <= prev_open and c_body > prev_body * 1.1:
                    patterns.append({
                        "date": date_str,
                        "index": i,
                        "pattern": "Bearish Engulfing",
                        "bias": "Strong Bearish Reversal",
                        "price": round(float(c_close), 2),
                        "significance": "Institutional distribution completely eclipses previous bullish advance.",
                        "failure_risk": "Reversal fails if subsequent session reclaims upper high."
                    })

            # 8. Piercing Line (In Downtrend: prev red, current green opens below prev low and closes > 50% into prev body)
            if prior_trend == "Downtrend" and not prev_bullish and is_bullish:
                if c_open < lows[i-1] and c_close > (prev_open + prev_close) / 2.0 and c_close < prev_open:
                    patterns.append({
                        "date": date_str,
                        "index": i,
                        "pattern": "Piercing Line",
                        "bias": "Bullish Reversal",
                        "price": round(float(c_close), 2),
                        "significance": "Bullish thrust recovering more than 50% of previous sell-off day.",
                        "failure_risk": "Subject to whipsaw in strongly trending bear markets."
                    })

            # 9. Dark Cloud Cover (In Uptrend: prev green, current red opens above prev high and closes > 50% into prev body)
            if prior_trend == "Uptrend" and prev_bullish and not is_bullish:
                if c_open > highs[i-1] and c_close < (prev_open + prev_close) / 2.0 and c_close > prev_open:
                    patterns.append({
                        "date": date_str,
                        "index": i,
                        "pattern": "Dark Cloud Cover",
                        "bias": "Bearish Reversal",
                        "price": round(float(c_close), 2),
                        "significance": "Opening gap higher completely rejected, closing deeply into prior body.",
                        "failure_risk": "Requires confirmation bearish session."
                    })

            # Triple Candle Patterns (i, i-1, i-2)
            if i >= 3:
                p2_open, p2_close = opens[i-2], closes[i-2]
                p2_body = abs(p2_close - p2_open)
                p2_bullish = p2_close >= p2_open
                p1_body = prev_body

                # 10. Morning Star (Long red, small body gap down, strong green closing > midpoint of candle i-2)
                if not p2_bullish and p2_body >= 0.01 * p2_open and p1_body < p2_body * 0.4:
                    if is_bullish and c_close > (p2_open + p2_close) / 2.0:
                        patterns.append({
                            "date": date_str,
                            "index": i,
                            "pattern": "Morning Star",
                            "bias": "High-Probability Bullish Reversal",
                            "price": round(float(c_close), 2),
                            "significance": "Classic 3-phase institutional bottoming formation: Panic -> Indecision -> Accumulation.",
                            "failure_risk": "Breach of the star's lower shadow invalidates setup."
                        })

                # 11. Evening Star (Long green, small body gap up, strong red closing > midpoint of candle i-2)
                if p2_bullish and p2_body >= 0.01 * p2_open and p1_body < p2_body * 0.4:
                    if not is_bullish and c_close < (p2_open + p2_close) / 2.0:
                        patterns.append({
                            "date": date_str,
                            "index": i,
                            "pattern": "Evening Star",
                            "bias": "High-Probability Bearish Reversal",
                            "price": round(float(c_close), 2),
                            "significance": "Classic 3-phase topping exhaustion: Euphoria -> Stall -> Distribution.",
                            "failure_risk": "Breakout above star high invalidates bearish pattern."
                        })

        return patterns


# Global candlestick pattern detector instance
candlestick_detector = CandlestickPatternDetector()
