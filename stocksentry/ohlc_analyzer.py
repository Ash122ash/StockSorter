"""
StockSentry AI - OHLC & Price Performance Analysis Engine
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def compute_ohlc_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes rigorous OHLC metrics and returns analysis on daily, 5-session, and 20-session returns.
    df is assumed to be sorted chronologically with Open, High, Low, Close, Volume.
    """
    if df is None or len(df) < 2:
        return {"error": "Insufficient data (minimum 2 sessions required for return analysis)"}

    # Clone and prepare
    d = df.copy()

    # Calculations
    d["Prev_Close"] = d["Close"].shift(1)
    d["Daily_Return_Pct"] = ((d["Close"] / d["Prev_Close"]) - 1.0) * 100.0
    d["Opening_Gap_Pct"] = ((d["Open"] / d["Prev_Close"]) - 1.0) * 100.0
    d["Intraday_Range_Pct"] = ((d["High"] - d["Low"]) / d["Low"]) * 100.0
    d["Intraday_Range_Abs"] = d["High"] - d["Low"]
    d["Vol_MA20"] = d["Volume"].rolling(window=min(20, len(d)), min_periods=1).mean()
    d["RVOL"] = d["Volume"] / d["Vol_MA20"].replace(0, np.nan)
    d["Vol_Change_Pct"] = ((d["Volume"] / d["Volume"].shift(1)) - 1.0) * 100.0

    # Max Drawdown calculation
    running_max = d["Close"].cummax()
    drawdowns = ((d["Close"] - running_max) / running_max) * 100.0
    max_drawdown = float(drawdowns.min())

    # Historical annualized volatility based on daily log returns
    log_returns = np.log(d["Close"] / d["Prev_Close"]).dropna()
    ann_volatility = float(log_returns.std() * np.sqrt(252) * 100.0) if len(log_returns) > 1 else 0.0

    # Summary metrics for latest session
    latest = d.iloc[-1]
    prev = d.iloc[-2]

    # Trading-session returns (not calendar days)
    # Five trading sessions return
    five_session_return = None
    if len(d) >= 6:
        close_5_ago = d["Close"].iloc[-6]
        five_session_return = round(((latest["Close"] / close_5_ago) - 1.0) * 100.0, 2)

    # Twenty trading sessions return (~ 1 month)
    twenty_session_return = None
    if len(d) >= 21:
        close_20_ago = d["Close"].iloc[-21]
        twenty_session_return = round(((latest["Close"] / close_20_ago) - 1.0) * 100.0, 2)
    elif len(d) > 1:
        twenty_session_return = round(((latest["Close"] / d["Close"].iloc[0]) - 1.0) * 100.0, 2)

    # Cumulative return over full dataset
    cumulative_return = round(((latest["Close"] / d["Close"].iloc[0]) - 1.0) * 100.0, 2)

    # Up vs Down sessions count
    returns_series = d["Daily_Return_Pct"].dropna()
    positive_sessions = int((returns_series > 0).sum())
    negative_sessions = int((returns_series < 0).sum())
    flat_sessions = int((returns_series == 0).sum())

    largest_rise = float(returns_series.max()) if not returns_series.empty else 0.0
    largest_fall = float(returns_series.min()) if not returns_series.empty else 0.0

    # Format recent sessions table (last 20 sessions for inspection)
    recent_records: List[Dict[str, Any]] = []
    tail_df = d.tail(30).iloc[::-1]  # reverse chronological
    for idx, row in tail_df.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        recent_records.append({
            "date": date_str,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
            "daily_return_pct": round(float(row["Daily_Return_Pct"]), 2) if pd.notnull(row["Daily_Return_Pct"]) else 0.0,
            "opening_gap_pct": round(float(row["Opening_Gap_Pct"]), 2) if pd.notnull(row["Opening_Gap_Pct"]) else 0.0,
            "intraday_range_pct": round(float(row["Intraday_Range_Pct"]), 2) if pd.notnull(row["Intraday_Range_Pct"]) else 0.0,
            "rvol": round(float(row["RVOL"]), 2) if pd.notnull(row["RVOL"]) else 1.0
        })

    return {
        "latest_close": round(float(latest["Close"]), 2),
        "latest_open": round(float(latest["Open"]), 2),
        "latest_high": round(float(latest["High"]), 2),
        "latest_low": round(float(latest["Low"]), 2),
        "latest_volume": int(latest["Volume"]),
        "previous_close": round(float(prev["Close"]), 2),
        "daily_change_abs": round(float(latest["Close"] - prev["Close"]), 2),
        "daily_return_pct": round(float(latest["Daily_Return_Pct"]), 2) if pd.notnull(latest["Daily_Return_Pct"]) else 0.0,
        "opening_gap_pct": round(float(latest["Opening_Gap_Pct"]), 2) if pd.notnull(latest["Opening_Gap_Pct"]) else 0.0,
        "intraday_range_pct": round(float(latest["Intraday_Range_Pct"]), 2),
        "intraday_range_abs": round(float(latest["Intraday_Range_Abs"]), 2),
        "five_session_return_pct": five_session_return,
        "twenty_session_return_pct": twenty_session_return,
        "cumulative_return_pct": cumulative_return,
        "avg_volume_20d": int(latest["Vol_MA20"]) if pd.notnull(latest["Vol_MA20"]) else int(latest["Volume"]),
        "relative_volume": round(float(latest["RVOL"]), 2) if pd.notnull(latest["RVOL"]) else 1.0,
        "volume_change_pct": round(float(latest["Vol_Change_Pct"]), 2) if pd.notnull(latest["Vol_Change_Pct"]) else 0.0,
        "positive_sessions": positive_sessions,
        "negative_sessions": negative_sessions,
        "flat_sessions": flat_sessions,
        "largest_daily_rise_pct": round(largest_rise, 2),
        "largest_daily_fall_pct": round(largest_fall, 2),
        "historical_annualized_volatility_pct": round(ann_volatility, 2),
        "maximum_drawdown_pct": round(max_drawdown, 2),
        "recent_ohlc_table": recent_records
    }
