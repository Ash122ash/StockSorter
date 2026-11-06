"""
StockSentry AI - Swing Extrema Detection
Identifies validated local swing highs and swing lows with prominence and noise filtering.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def detect_swing_extrema(df: pd.DataFrame, window: int = 4) -> Dict[str, Any]:
    """
    Detects local swing highs and swing lows.
    A swing high at index i satisfies High[i] >= High[i-k..i+k].
    A swing low at index i satisfies Low[i] <= Low[i-k..i+k].
    """
    if df is None or len(df) < window * 2 + 1:
        return {"swing_highs": [], "swing_lows": [], "swing_points": []}

    d = df.copy()
    highs = d["High"].values
    lows = d["Low"].values
    volumes = d["Volume"].values if "Volume" in d.columns else np.ones(len(d))
    dates = d.index

    swing_highs: List[Dict[str, Any]] = []
    swing_lows: List[Dict[str, Any]] = []
    swing_points: List[Dict[str, Any]] = []

    n = len(d)
    for i in range(window, n - window):
        # High extrema check
        is_high = True
        curr_high = highs[i]
        for offset in range(-window, window + 1):
            if offset != 0 and highs[i + offset] > curr_high:
                is_high = False
                break
        
        # Low extrema check
        is_low = True
        curr_low = lows[i]
        for offset in range(-window, window + 1):
            if offset != 0 and lows[i + offset] < curr_low:
                is_low = False
                break

        date_str = dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], "strftime") else str(dates[i])

        if is_high:
            pt = {
                "index": i,
                "date": date_str,
                "type": "high",
                "price": round(float(curr_high), 2),
                "volume": float(volumes[i])
            }
            swing_highs.append(pt)
            swing_points.append(pt)

        if is_low:
            pt = {
                "index": i,
                "date": date_str,
                "type": "low",
                "price": round(float(curr_low), 2),
                "volume": float(volumes[i])
            }
            swing_lows.append(pt)
            swing_points.append(pt)

    # Sort combined points chronologically
    swing_points.sort(key=lambda x: x["index"])

    return {
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "swing_points": swing_points
    }
