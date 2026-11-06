"""
StockSentry AI - Professional Technical Analysis Engine
Calculates standard technical indicators strictly on consistent price series (Close).
Provides educational explanations and market structure analysis (HH/HL, LH/LL, Gaps).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Calculates all key trading and momentum indicators with zero external opaque C-bindings."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculates all indicators on unadjusted/consistent Close price series.
        Returns (augmented_dataframe, indicators_summary_dict).
        """
        if df is None or len(df) < 14:
            return df, {"error": "Insufficient history for technical indicator suite (need at least 14 sessions)"}

        d = df.copy()
        close = d["Close"]
        high = d["High"]
        low = d["Low"]
        volume = d["Volume"]

        # 1. Moving Averages
        d["SMA_5"] = close.rolling(window=5, min_periods=1).mean()
        d["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
        d["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
        d["EMA_9"] = close.ewm(span=9, adjust=False).mean()
        d["EMA_21"] = close.ewm(span=21, adjust=False).mean()

        # 2. RSI (14) - Wilder's smoothed method
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        d["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))
        d["RSI_14"] = d["RSI_14"].fillna(50.0)

        # 3. MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        d["MACD_Line"] = ema_12 - ema_26
        d["MACD_Signal"] = d["MACD_Line"].ewm(span=9, adjust=False).mean()
        d["MACD_Hist"] = d["MACD_Line"] - d["MACD_Signal"]

        # 4. Bollinger Bands (20, 2)
        bb_mid = close.rolling(window=20, min_periods=1).mean()
        bb_std = close.rolling(window=20, min_periods=1).std().fillna(0)
        d["BB_Middle"] = bb_mid
        d["BB_Upper"] = bb_mid + (2.0 * bb_std)
        d["BB_Lower"] = bb_mid - (2.0 * bb_std)
        d["BB_Width_Pct"] = ((d["BB_Upper"] - d["BB_Lower"]) / d["BB_Middle"].replace(0, np.nan)) * 100.0

        # 5. ATR (14)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        d["ATR_14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().bfill()

        # 6. Volume & Momentum
        d["Vol_SMA_20"] = volume.rolling(window=20, min_periods=1).mean()
        d["RVOL"] = volume / d["Vol_SMA_20"].replace(0, np.nan)
        d["Momentum_10"] = close - close.shift(10)
        d["ROC_10"] = ((close / close.shift(10)) - 1.0) * 100.0

        # 7. 20-day Historical Volatility (annualized)
        log_ret = np.log(close / close.shift(1))
        d["HV_20"] = log_ret.rolling(window=20, min_periods=5).std() * np.sqrt(252) * 100.0

        # 8. Market Structure: Swing Highs & Lows (5-bar local extrema)
        d["Swing_High"] = (high == high.rolling(window=5, center=True).max())
        d["Swing_Low"] = (low == low.rolling(window=5, center=True).min())

        # 9. Price Gaps detection (unfilled or material gaps > 0.75%)
        gap_up = (low > prev_close * 1.0075)
        gap_down = (high < prev_close * 0.9925)
        recent_gaps = []
        for idx in d[gap_up | gap_down].tail(5).index:
            row = d.loc[idx]
            is_up = bool(row["Open"] > d.loc[:idx]["Close"].iloc[-2]) if len(d.loc[:idx]) > 1 else False
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            recent_gaps.append({
                "date": date_str,
                "type": "Gap Up" if is_up else "Gap Down",
                "gap_pct": round(float(((row["Open"] / d.loc[:idx]["Close"].iloc[-2]) - 1.0) * 100.0), 2) if len(d.loc[:idx]) > 1 else 0.0
            })

        # 10. Trend Direction Evaluation via Market Structure & Moving Averages
        latest = d.iloc[-1]
        c_price = float(latest["Close"])
        sma20 = float(latest["SMA_20"])
        sma50 = float(latest["SMA_50"])
        ema9 = float(latest["EMA_9"])
        ema21 = float(latest["EMA_21"])

        # Identify recent swings
        swing_high_series = d[d["Swing_High"]]["High"]
        swing_low_series = d[d["Swing_Low"]]["Low"]

        structure_trend = "Consolidation / Range"
        if len(swing_high_series) >= 2 and len(swing_low_series) >= 2:
            last_sh1, last_sh2 = swing_high_series.iloc[-1], swing_high_series.iloc[-2]
            last_sl1, last_sl2 = swing_low_series.iloc[-1], swing_low_series.iloc[-2]
            if last_sh1 > last_sh2 and last_sl1 > last_sl2:
                structure_trend = "Higher Highs & Higher Lows (Uptrend Structure)"
            elif last_sh1 < last_sh2 and last_sl1 < last_sl2:
                structure_trend = "Lower Highs & Lower Lows (Downtrend Structure)"

        # Composite Trend
        bullish_votes = 0
        bearish_votes = 0

        if c_price > sma20: bullish_votes += 1
        else: bearish_votes += 1

        if ema9 > ema21: bullish_votes += 1
        else: bearish_votes += 1

        if c_price > sma50: bullish_votes += 1
        else: bearish_votes += 1

        if latest["MACD_Hist"] > 0: bullish_votes += 1
        else: bearish_votes += 1

        if latest["RSI_14"] > 50: bullish_votes += 1
        else: bearish_votes += 1

        if bullish_votes >= 4:
            primary_trend = "Strong Bullish"
        elif bullish_votes == 3:
            primary_trend = "Mildly Bullish / Neutral"
        elif bearish_votes >= 4:
            primary_trend = "Strong Bearish"
        elif bearish_votes == 3:
            primary_trend = "Mildly Bearish / Neutral"
        else:
            primary_trend = "Neutral / Sideways"

        summary = {
            "price_basis": "Unadjusted daily close prices across all technical metrics for consistency",
            "primary_trend": primary_trend,
            "market_structure": structure_trend,
            "indicators": {
                "SMA_5": round(float(latest["SMA_5"]), 2),
                "SMA_20": round(float(latest["SMA_20"]), 2),
                "SMA_50": round(float(latest["SMA_50"]), 2),
                "EMA_9": round(float(latest["EMA_9"]), 2),
                "EMA_21": round(float(latest["EMA_21"]), 2),
                "RSI_14": round(float(latest["RSI_14"]), 2),
                "MACD_Line": round(float(latest["MACD_Line"]), 2),
                "MACD_Signal": round(float(latest["MACD_Signal"]), 2),
                "MACD_Hist": round(float(latest["MACD_Hist"]), 2),
                "BB_Upper": round(float(latest["BB_Upper"]), 2),
                "BB_Middle": round(float(latest["BB_Middle"]), 2),
                "BB_Lower": round(float(latest["BB_Lower"]), 2),
                "BB_Width_Pct": round(float(latest["BB_Width_Pct"]), 2),
                "ATR_14": round(float(latest["ATR_14"]), 2),
                "Momentum_10": round(float(latest["Momentum_10"]), 2) if pd.notnull(latest["Momentum_10"]) else 0.0,
                "ROC_10_Pct": round(float(latest["ROC_10"]), 2) if pd.notnull(latest["ROC_10"]) else 0.0,
                "RVOL": round(float(latest["RVOL"]), 2) if pd.notnull(latest["RVOL"]) else 1.0,
                "HV_20_Pct": round(float(latest["HV_20"]), 2) if pd.notnull(latest["HV_20"]) else 0.0,
            },
            "recent_gaps": recent_gaps,
            "explanations": {
                "SMA_EMA": "Moving average stack (9 EMA, 21 EMA, 50 SMA) signals trend hierarchy. Price above all moving averages indicates sustained institutional accumulation.",
                "RSI_14": "Relative Strength Index evaluates velocity of directional price moves. Overbought (>70) and oversold (<30) indicate momentum extremes, while 50 marks bull/bear equilibrium.",
                "MACD": "Moving Average Convergence Divergence tracks trend shifts and momentum acceleration. Expanding positive histograms show rising buyer dominance.",
                "Bollinger_Bands": "Envelopes price within 2 standard deviations of a 20-period baseline. Band squeezes precede volatility expansion; band walks signal trend continuation.",
                "ATR_14": "Average True Range gauges non-directional volatility and typical session movement, used for setting dynamic stop-loss margins and zone tolerance.",
                "RVOL": "Relative Volume compares current volume to the 20-day average. Spikes above 1.5x validate trend breakouts or capitulation sells."
            }
        }

        return d, summary
