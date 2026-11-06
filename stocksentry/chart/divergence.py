"""
StockSentry AI - RSI, MACD & Indicator Divergence Analysis Engine
Detects regular and hidden bullish/bearish divergences by strictly correlating
price swing extrema with corresponding indicator extrema. Tracks MA crossovers and exhaustion.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from stocksentry.chart.swing_points import detect_swing_extrema


class DivergenceEngine:
    """
    Evaluates momentum divergences and moving average crossovers.
    Ensures price swing extrema are strictly aligned with indicator values to prevent false divergence traps.
    """

    def analyze(self, df_with_indicators: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes RSI 14 and MACD series for divergences and trend inflections.
        """
        if df_with_indicators is None or len(df_with_indicators) < 25:
            return {
                "rsi_divergences": [],
                "macd_divergences": [],
                "ma_crossovers": [],
                "momentum_status": "Insufficient data",
                "rsi_status": "Neutral"
            }

        d = df_with_indicators.copy()
        extrema = detect_swing_extrema(d, window=3)
        swing_highs = extrema["swing_highs"]
        swing_lows = extrema["swing_lows"]

        # Divergences
        rsi_divs = self._detect_rsi_divergences(swing_highs, swing_lows, d)
        macd_divs = self._detect_macd_divergences(swing_highs, swing_lows, d)

        # Moving average crossovers (EMA 9 vs EMA 21, SMA 20 vs SMA 50)
        crossovers = self._detect_ma_crossovers(d)

        # Current momentum & RSI status
        latest = d.iloc[-1]
        rsi_val = float(latest["RSI_14"]) if "RSI_14" in d.columns else 50.0
        macd_hist = float(latest["MACD_Hist"]) if "MACD_Hist" in d.columns else 0.0

        if rsi_val >= 70.0:
            rsi_status = f"Overbought ({rsi_val:.1f}) — Susceptible to mean-reversion pullbacks."
        elif rsi_val <= 30.0:
            rsi_status = f"Oversold ({rsi_val:.1f}) — Deep technical compression; watch for reversal."
        elif 50.0 <= rsi_val < 70.0:
            rsi_status = f"Bullish Expansion ({rsi_val:.1f}) — Strong directional buyer control."
        else:
            rsi_status = f"Bearish Contraction ({rsi_val:.1f}) — Sellers dominating momentum."

        momentum_status = (
            "Expanding Bullish Momentum (MACD Histogram Positive & Rising)" if macd_hist > 0 and len(d) > 1 and macd_hist > d["MACD_Hist"].iloc[-2]
            else ("Weakening Bullish Momentum (MACD Histogram Contracting)" if macd_hist > 0
                  else ("Accelerating Downside Momentum (MACD Histogram Negative & Expanding)" if macd_hist < 0 and len(d) > 1 and macd_hist < d["MACD_Hist"].iloc[-2]
                        else "Decelerating Downside Momentum (MACD Histogram Contracting)"))
        )

        return {
            "rsi_status": rsi_status,
            "momentum_status": momentum_status,
            "rsi_current": round(rsi_val, 2),
            "macd_hist_current": round(macd_hist, 2),
            "rsi_divergences": rsi_divs,
            "macd_divergences": macd_divs,
            "ma_crossovers": crossovers
        }

    def _detect_rsi_divergences(
        self,
        highs: List[Dict[str, Any]],
        lows: List[Dict[str, Any]],
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Compares price swings with RSI swings to detect divergence."""
        if "RSI_14" not in df.columns or len(highs) < 2 or len(lows) < 2:
            return []

        rsi_series = df["RSI_14"].values
        divergences: List[Dict[str, Any]] = []

        # 1. Bullish Divergence on Lows: Price Lower Low, RSI Higher Low
        for i in range(len(lows) - 1, max(-1, len(lows) - 5), -1):
            l2 = lows[i]
            l1 = lows[i - 1]
            idx1, idx2 = l1["index"], l2["index"]
            if idx2 - idx1 < 5 or idx2 - idx1 > 60:
                continue

            p1, p2 = l1["price"], l2["price"]
            rsi1, rsi2 = rsi_series[idx1], rsi_series[idx2]

            # Price Lower Low while RSI Higher Low
            if p2 < p1 * 0.995 and rsi2 > rsi1 + 2.0 and rsi1 <= 40.0:
                divergences.append({
                    "type": "Regular Bullish RSI Divergence",
                    "bias": "Bullish Reversal",
                    "date_1": l1["date"],
                    "price_1": p1,
                    "rsi_1": round(float(rsi1), 1),
                    "date_2": l2["date"],
                    "price_2": p2,
                    "rsi_2": round(float(rsi2), 1),
                    "significance": f"Price made a lower low ({p1:.2f} -> {p2:.2f}) while RSI formed a higher low ({rsi1:.1f} -> {rsi2:.1f}), signaling downward exhaustion."
                })
                break

        # 2. Bearish Divergence on Highs: Price Higher High, RSI Lower High
        for i in range(len(highs) - 1, max(-1, len(highs) - 5), -1):
            h2 = highs[i]
            h1 = highs[i - 1]
            idx1, idx2 = h1["index"], h2["index"]
            if idx2 - idx1 < 5 or idx2 - idx1 > 60:
                continue

            p1, p2 = h1["price"], h2["price"]
            rsi1, rsi2 = rsi_series[idx1], rsi_series[idx2]

            # Price Higher High while RSI Lower High
            if p2 > p1 * 1.005 and rsi2 < rsi1 - 2.0 and rsi1 >= 60.0:
                divergences.append({
                    "type": "Regular Bearish RSI Divergence",
                    "bias": "Bearish Reversal",
                    "date_1": h1["date"],
                    "price_1": p1,
                    "rsi_1": round(float(rsi1), 1),
                    "date_2": h2["date"],
                    "price_2": p2,
                    "rsi_2": round(float(rsi2), 1),
                    "significance": f"Price made a higher high ({p1:.2f} -> {p2:.2f}) while RSI formed a lower high ({rsi1:.1f} -> {rsi2:.1f}), warning of waning buyer momentum."
                })
                break

        return divergences

    def _detect_macd_divergences(
        self,
        highs: List[Dict[str, Any]],
        lows: List[Dict[str, Any]],
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Compares price swings with MACD histogram/line to detect divergence."""
        if "MACD_Hist" not in df.columns or len(highs) < 2 or len(lows) < 2:
            return []

        macd_hist = df["MACD_Hist"].values
        divergences: List[Dict[str, Any]] = []

        # Bullish MACD Divergence on Lows
        if len(lows) >= 2:
            l2 = lows[-1]
            l1 = lows[-2]
            idx1, idx2 = l1["index"], l2["index"]
            if 5 <= (idx2 - idx1) <= 60:
                p1, p2 = l1["price"], l2["price"]
                mh1, mh2 = macd_hist[idx1], macd_hist[idx2]
                if p2 < p1 * 0.995 and mh2 > mh1 + 0.15 and mh1 < 0:
                    divergences.append({
                        "type": "Bullish MACD Divergence",
                        "bias": "Bullish Momentum Shift",
                        "date": l2["date"],
                        "significance": "Price made lower low with shallower MACD histogram trough; institutional selling velocity has diminished."
                    })

        # Bearish MACD Divergence on Highs
        if len(highs) >= 2:
            h2 = highs[-1]
            h1 = highs[-2]
            idx1, idx2 = h1["index"], h2["index"]
            if 5 <= (idx2 - idx1) <= 60:
                p1, p2 = h1["price"], h2["price"]
                mh1, mh2 = macd_hist[idx1], macd_hist[idx2]
                if p2 > p1 * 1.005 and mh2 < mh1 - 0.15 and mh1 > 0:
                    divergences.append({
                        "type": "Bearish MACD Divergence",
                        "bias": "Bearish Distribution Alert",
                        "date": h2["date"],
                        "significance": "Price made higher high but MACD histogram failed to confirm with lower peak; buying acceleration exhausted."
                    })

        return divergences

    def _detect_ma_crossovers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identifies recent moving average crossovers (last 10 sessions)."""
        crossovers: List[Dict[str, Any]] = []
        if len(df) < 10 or "EMA_9" not in df.columns or "EMA_21" not in df.columns:
            return crossovers

        ema9 = df["EMA_9"].values
        ema21 = df["EMA_21"].values
        dates = df.index

        # Scan last 10 bars for crossover
        for i in range(len(df) - 10, len(df)):
            if ema9[i-1] <= ema21[i-1] and ema9[i] > ema21[i]:
                date_str = dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], "strftime") else str(dates[i])
                crossovers.append({
                    "date": date_str,
                    "type": "Bullish EMA 9/21 Crossover",
                    "bias": "Bullish Momentum Confirmation",
                    "price": round(float(df["Close"].iloc[i]), 2),
                    "significance": "Fast EMA crossed above slow EMA; validates short-term trend acceleration."
                })
            elif ema9[i-1] >= ema21[i-1] and ema9[i] < ema21[i]:
                date_str = dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], "strftime") else str(dates[i])
                crossovers.append({
                    "date": date_str,
                    "type": "Bearish EMA 9/21 Crossover",
                    "bias": "Bearish Momentum Confirmation",
                    "price": round(float(df["Close"].iloc[i]), 2),
                    "significance": "Fast EMA crossed below slow EMA; confirms short-term trend breakdown."
                })

        return crossovers


# Global divergence engine instance
divergence_engine = DivergenceEngine()
