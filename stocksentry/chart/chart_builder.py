"""
StockSentry AI - Advanced Interactive Plotly Chart Builder
Constructs professional 3-pane synchronized financial charting workstation with
candlesticks, MAs, Bollinger Bands, S/R zones, trendlines, pattern markers, divergences,
and forward 5-session scenario cones.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_advanced_plotly_chart(
    df: pd.DataFrame,
    df_augmented: pd.DataFrame,
    sr_data: Dict[str, Any],
    trendline_data: Dict[str, Any],
    candlestick_patterns: List[Dict[str, Any]],
    chart_patterns: List[Dict[str, Any]],
    divergence_data: Dict[str, Any],
    forecast_scenarios: Dict[str, Any],
    symbol: str,
    currency: str = "INR"
) -> go.Figure:
    """
    Creates institutional Plotly chart with 3 synchronized subplots:
    Row 1: Candlesticks, MAs, Bollinger Bands, S/R zones, Trendlines, Pattern markers, Future Forecast Scenarios
    Row 2: Volume & 20D Volume MA
    Row 3: RSI 14 & MACD Histogram / Lines
    """
    curr = "₹" if currency == "INR" else "$"
    d = df_augmented.copy()
    dates = d.index

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(
            f"{symbol} — Price Action, Liquidity Zones & Structural Patterns",
            "Trading Volume & RVOL Activity",
            "RSI (14) & Momentum Divergence Suite"
        )
    )

    # 1. Main Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=d["Open"],
            high=d["High"],
            low=d["Low"],
            close=d["Close"],
            name="OHLC",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#10b981",
            decreasing_fillcolor="#ef4444"
        ),
        row=1, col=1
    )

    # 2. Moving Average Overlays
    if "EMA_9" in d.columns:
        fig.add_trace(go.Scatter(x=dates, y=d["EMA_9"], name="EMA 9", line=dict(color="#38bdf8", width=1.2)), row=1, col=1)
    if "EMA_21" in d.columns:
        fig.add_trace(go.Scatter(x=dates, y=d["EMA_21"], name="EMA 21", line=dict(color="#fbbf24", width=1.2)), row=1, col=1)
    if "SMA_20" in d.columns:
        fig.add_trace(go.Scatter(x=dates, y=d["SMA_20"], name="SMA 20", line=dict(color="#60a5fa", width=1.0, dash="dot")), row=1, col=1)
    if "SMA_50" in d.columns:
        fig.add_trace(go.Scatter(x=dates, y=d["SMA_50"], name="SMA 50", line=dict(color="#a855f7", width=1.5)), row=1, col=1)

    # 3. Bollinger Bands (20, 2)
    if "BB_Upper" in d.columns and "BB_Lower" in d.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=d["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(148, 163, 184, 0.4)", dash="dash", width=0.9),
            hoverinfo="skip"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=d["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(148, 163, 184, 0.4)", dash="dash", width=0.9),
            fill='tonexty', fillcolor='rgba(148, 163, 184, 0.05)',
            hoverinfo="skip"
        ), row=1, col=1)

    # 4. Support Zones (Shaded Green Horizontal Bands)
    for zone in sr_data.get("support_zones", [])[:3]:
        fig.add_hrect(
            y0=zone["lower_boundary"], y1=zone["upper_boundary"],
            fillcolor="rgba(16, 185, 129, 0.12)", line_width=1, line_color="#10b981",
            annotation_text=f"Support: {curr}{zone['lower_boundary']}-{zone['upper_boundary']} ({zone['meaningful_tests']} tests)",
            annotation_position="top left",
            row=1, col=1
        )

    # 5. Resistance Zones (Shaded Red Horizontal Bands)
    for zone in sr_data.get("resistance_zones", [])[:3]:
        fig.add_hrect(
            y0=zone["lower_boundary"], y1=zone["upper_boundary"],
            fillcolor="rgba(239, 68, 68, 0.12)", line_width=1, line_color="#ef4444",
            annotation_text=f"Resistance: {curr}{zone['lower_boundary']}-{zone['upper_boundary']} ({zone['meaningful_tests']} tests)",
            annotation_position="bottom left",
            row=1, col=1
        )

    # 6. Trendlines Overlay
    for tl in trendline_data.get("trendlines", []):
        fig.add_trace(go.Scatter(
            x=[tl["start_date"], tl["end_date"]],
            y=[tl["start_price"], tl["end_price"]],
            mode="lines+text",
            name=tl["type"],
            line=dict(color="#38bdf8" if tl["is_support"] else "#f43f5e", width=1.8, dash="solid"),
            text=["", f"{tl['type']} ({tl['touches']} touches)"],
            textposition="top right"
        ), row=1, col=1)

    # 7. Candlestick Pattern Markers (Annotations)
    for cp in candlestick_patterns[-6:]: # Last 6 detected patterns
        is_bull = "Bullish" in cp["bias"]
        fig.add_annotation(
            x=cp["date"],
            y=cp["price"],
            text=f"🕯️ {cp['pattern']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#10b981" if is_bull else "#ef4444",
            ax=0,
            ay=-35 if is_bull else 35,
            bgcolor="#111827",
            bordercolor="#10b981" if is_bull else "#ef4444",
            borderwidth=1,
            font=dict(size=9, color="#f8fafc"),
            row=1, col=1
        )

    # 8. Future 5-Session Forecast Scenario Projections (Dashed lines plotted into future dates)
    fut_dates = forecast_scenarios.get("future_dates", [])
    if fut_dates:
        anchor_date = forecast_scenarios.get("anchor_date")
        anchor_price = forecast_scenarios.get("anchor_price")
        
        # Connect anchor to future path
        bull_x = [anchor_date] + fut_dates
        bull_y = [anchor_price] + forecast_scenarios["bullish_scenario"]["trajectory"]
        
        base_x = [anchor_date] + fut_dates
        base_y = [anchor_price] + forecast_scenarios["base_scenario"]["trajectory"]
        
        bear_x = [anchor_date] + fut_dates
        bear_y = [anchor_price] + forecast_scenarios["bearish_scenario"]["trajectory"]

        # Bullish Trajectory
        fig.add_trace(go.Scatter(
            x=bull_x, y=bull_y,
            name="Forecast: Bullish P90",
            line=dict(color="#10b981", width=2.0, dash="dash"),
            mode="lines+markers"
        ), row=1, col=1)

        # Base Trajectory
        fig.add_trace(go.Scatter(
            x=base_x, y=base_y,
            name="Forecast: Base P50",
            line=dict(color="#f59e0b", width=2.0, dash="dot"),
            mode="lines+markers"
        ), row=1, col=1)

        # Bearish Trajectory
        fig.add_trace(go.Scatter(
            x=bear_x, y=bear_y,
            name="Forecast: Bearish P10",
            line=dict(color="#ef4444", width=2.0, dash="dash"),
            mode="lines+markers"
        ), row=1, col=1)

        # Future Shaded Uncertainty Cone
        upper_cone = [anchor_price] + forecast_scenarios["uncertainty_cone"]["upper_band"]
        lower_cone = [anchor_price] + forecast_scenarios["uncertainty_cone"]["lower_band"]

        fig.add_trace(go.Scatter(
            x=bull_x, y=upper_cone,
            name="Forecast Upper Uncertainty",
            line=dict(color="rgba(59, 130, 246, 0.2)", width=0),
            hoverinfo="skip",
            showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=bull_x, y=lower_cone,
            name="Forecast Uncertainty Cone",
            line=dict(color="rgba(59, 130, 246, 0.2)", width=0),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.08)',
            hoverinfo="skip"
        ), row=1, col=1)

    # 9. Row 2: Volume Subplot & 20D MA
    vol_colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(d["Close"], d["Open"])]
    fig.add_trace(go.Bar(
        x=dates, y=d["Volume"], name="Volume", marker_color=vol_colors
    ), row=2, col=1)

    if "Vol_SMA_20" in d.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=d["Vol_SMA_20"], name="Vol MA 20",
            line=dict(color="#94a3b8", width=1.2)
        ), row=2, col=1)

    # 10. Row 3: RSI (14) & Overbought/Oversold Thresholds
    if "RSI_14" in d.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=d["RSI_14"], name="RSI (14)",
            line=dict(color="#f43f5e", width=1.5)
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#64748b", row=3, col=1)

        # Mark RSI Divergences
        for rdiv in divergence_data.get("rsi_divergences", []):
            fig.add_annotation(
                x=rdiv["date_2"],
                y=rdiv["rsi_2"],
                text="⚡ Div",
                showarrow=True,
                arrowhead=1,
                arrowcolor="#38bdf8",
                font=dict(size=8, color="#38bdf8"),
                row=3, col=1
            )

    # Styling layout
    fig.update_layout(
        height=850,
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#080c14",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )

    return fig
