"""
StockSentry AI — Advanced Automated Technical Chart Analysis & Crash-Risk Platform
Interactive Streamlit Dashboard with Plotly & Dark Financial-Terminal Interface.
Run:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from stocksentry.orchestrator import run_full_analysis
from stocksentry.chart.chart_builder import build_advanced_plotly_chart

st.set_page_config(
    page_title="StockSentry AI — Advanced Chart Analysis & Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark Terminal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #080c14;
        color: #e2e8f0;
    }
    
    .metric-card {
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 0.80rem;
        font-weight: 600;
        margin-top: 2px;
    }
    
    .qa-box {
        background: #131b2e;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .qa-q {
        font-weight: 700;
        color: #60a5fa;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    .qa-a {
        color: #e2e8f0;
        font-size: 0.90rem;
        line-height: 1.5;
    }
    
    .reading-box {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .reading-header {
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        color: #38bdf8;
        letter-spacing: 0.05em;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .reading-body {
        font-size: 0.92rem;
        color: #f1f5f9;
        line-height: 1.6;
    }
    
    .badge-bullish {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-bearish {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-neutral {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### 🛡️ **StockSentry AI**")
st.sidebar.markdown("<span style='font-size:0.8rem; color:#94a3b8;'>Advanced Automated Chart Analysis Workstation</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

quick_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "AAPL", "MSFT"]
selected_quick = st.sidebar.selectbox("Quick Tickers", ["Custom Symbol"] + quick_tickers, index=1)

if selected_quick != "Custom Symbol":
    symbol_input = selected_quick
else:
    symbol_input = st.sidebar.text_input("Enter Ticker Symbol", value="RELIANCE.NS")

period_choice = st.sidebar.selectbox(
    "Chart Historical Horizon",
    options=["1y", "2y", "6mo", "3mo", "1mo", "max"],
    index=1
)

force_refresh = st.sidebar.checkbox("Force Cache Refresh", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.75rem; color: #64748b;">
<b>Chart Engine:</b> Plotly 3-Pane Workstation<br>
<b>Pattern Algorithms:</b> Classical Geometry + Anatomy<br>
<b>Trendlines:</b> Automatic Multi-Touch Fitting<br>
<b>ML Validation:</b> Walk-Forward Chronological
</div>
""", unsafe_allow_html=True)

symbol = symbol_input.strip().upper()

# Run Analysis
with st.spinner(f"Scanning market structure, patterns, and scenarios for {symbol}..."):
    try:
        data = run_full_analysis(symbol, period=period_choice, force_refresh=force_refresh)
    except Exception as e:
        st.error(f"Analysis failed for '{symbol}': {str(e)}")
        st.stop()

meta = data["metadata"]
ohlc = data["ohlc_metrics"]
tech = data["technical_analysis"]
sr = data["support_resistance"]
news = data["news_intelligence"]
forecast = data["forecasting"]
crash = data["crash_risk"]
val = data["walk_forward_validation"]
qa = data["questions_and_answers"]
trendlines = data["trendline_analysis"]
chart_patterns = data["chart_patterns"]
candlestick_patterns = data["candlestick_patterns"]
divergence = data["divergence_analysis"]
forecast_scenarios = data["forecast_scenarios"]
signal_backtest = data["signal_backtesting"]
reading = data["technical_analyst_reading"]

currency = "₹" if meta["currency"] == "INR" else "$"

# Top Banner Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    ret_color = '#10b981' if ohlc['daily_return_pct'] >= 0 else '#ef4444'
    st.markdown(f"## **{symbol}** &nbsp; <span style='font-size:2rem; font-family:JetBrains Mono;'>{currency}{ohlc['latest_close']:.2f}</span> &nbsp; "
                f"<span style='font-size:1.15rem; color:{ret_color};'>"
                f"{'+' if ohlc['daily_return_pct']>=0 else ''}{ohlc['daily_return_pct']:.2f}% ({'+' if ohlc['daily_change_abs']>=0 else ''}{ohlc['daily_change_abs']:.2f})</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#94a3b8; font-size:0.85rem;'>Latest Session: <b>{meta['latest_session']}</b> • Primary Trend: <b>{trendlines.get('current_trend')}</b> • Structure: <b>{trendlines.get('market_structure')}</b> • {meta['data_disclaimer']}</span>", unsafe_allow_html=True)

with col_h2:
    outlook_code = forecast.get("outlook_code", "NEUTRAL")
    badge_class = "badge-bullish" if outlook_code == "BULLISH" else ("badge-bearish" if outlook_code == "BEARISH" else "badge-neutral")
    st.markdown(f"<div style='text-align:right; margin-top:8px;'>"
                f"<span class='{badge_class}' style='font-size:1rem; padding:8px 14px;'>{forecast.get('outlook', 'Neutral')}</span><br>"
                f"<span style='font-size:0.75rem; color:#94a3b8;'>Crash Risk: <b style='color:{crash.get('risk_color')};'>{crash.get('risk_category')} ({crash.get('risk_score_100')}/100)</b></span>"
                f"</div>", unsafe_allow_html=True)

st.markdown("---")

# Key Metrics Grid
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    ret5 = ohlc.get("five_session_return_pct", 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">5-Session Return</div>
        <div class="metric-value" style="color:{'#10b981' if ret5 and ret5>=0 else '#ef4444'};">{'+' if ret5 and ret5>=0 else ''}{ret5}%</div>
        <div class="metric-delta">Past 1 Week</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    ret20 = ohlc.get("twenty_session_return_pct", 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">20-Session Return</div>
        <div class="metric-value" style="color:{'#10b981' if ret20 and ret20>=0 else '#ef4444'};">{'+' if ret20 and ret20>=0 else ''}{ret20}%</div>
        <div class="metric-delta">Past 1 Month</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    gap = ohlc.get("opening_gap_pct", 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Opening Gap</div>
        <div class="metric-value" style="color:{'#10b981' if gap and gap>=0 else '#ef4444'};">{'+' if gap and gap>=0 else ''}{gap}%</div>
        <div class="metric-delta">Open vs Prior Close</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    rvol = ohlc.get("relative_volume", 1.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RVOL (20D)</div>
        <div class="metric-value" style="color:{'#f59e0b' if rvol>1.5 else '#e2e8f0'};">{rvol}x</div>
        <div class="metric-delta">Volume Activity</div>
    </div>
    """, unsafe_allow_html=True)
with m5:
    hv = ohlc.get("historical_annualized_volatility_pct", 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Ann. Volatility</div>
        <div class="metric-value">{hv}%</div>
        <div class="metric-delta">Realized 20-Day</div>
    </div>
    """, unsafe_allow_html=True)
with m6:
    mdd = ohlc.get("maximum_drawdown_pct", 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max Drawdown</div>
        <div class="metric-value" style="color:#ef4444;">{mdd}%</div>
        <div class="metric-delta">Historical Peak</div>
    </div>
    """, unsafe_allow_html=True)

# Main Navigation Tabs
tab_chart, tab_reading, tab_trend, tab_patterns, tab_candlestick, tab_scenarios, tab_sr, tab_qa, tab_backtest, tab_news = st.tabs([
    "📈 Advanced Interactive Chart",
    "🧠 Technical Analyst's Reading",
    "📐 Trendlines & Channels",
    "🏛️ Classical Chart Patterns",
    "🕯️ Candlestick Patterns",
    "🔮 5-Day Forecast Scenarios",
    "🧱 Support & Resistance",
    "❓ 11 Core Questions",
    "🧪 Signal Backtesting Matrix",
    "📰 News Intelligence"
])

# -----------------------------------------------------------------------------
# TAB 1: ADVANCED INTERACTIVE PLOTLY CHART
# -----------------------------------------------------------------------------
with tab_chart:
    st.markdown("#### **Professional 3-Pane Technical Charting Workstation**")
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>Features Candlesticks, Moving Averages (9/21 EMA, 20/50 SMA), Bollinger Bands, Shaded S/R Zones, Trendlines, Candlestick Markers, Future Scenario Projections, Volume, and RSI (14) with Divergences.</p>", unsafe_allow_html=True)

    # Reconstruct dataframes for Plotly builder
    df_raw = pd.DataFrame(data["chart_data"])
    if not df_raw.empty:
        df_raw["Date"] = pd.to_datetime(df_raw["date"])
        df_raw.set_index("Date", inplace=True)
        # Capitalize for chart builder
        df_raw.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
            "ema9": "EMA_9", "ema21": "EMA_21", "sma20": "SMA_20", "sma50": "SMA_50",
            "bb_upper": "BB_Upper", "bb_lower": "BB_Lower", "rsi14": "RSI_14", "macd_hist": "MACD_Hist"
        }, inplace=True)

        fig = build_advanced_plotly_chart(
            df=df_raw,
            df_augmented=df_raw,
            sr_data=sr,
            trendline_data=trendlines,
            candlestick_patterns=candlestick_patterns,
            chart_patterns=chart_patterns,
            divergence_data=divergence,
            forecast_scenarios=forecast_scenarios,
            symbol=symbol,
            currency=meta.get("currency", "INR")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient chart data available.")


# -----------------------------------------------------------------------------
# TAB 2: TECHNICAL ANALYST'S READING (SYNTHESIZED DETERMINISTICALLY)
# -----------------------------------------------------------------------------
with tab_reading:
    st.markdown("#### **Technical Analyst's Reading — Automated Synthesis**")
    st.caption(reading["methodology_declaration"])

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(f"""
        <div class="reading-box" style="border-left:4px solid #38bdf8;">
            <div class="reading-header">Current Trend & Market Structure</div>
            <div class="reading-body">{reading['current_trend_analysis']}</div>
        </div>
        <div class="reading-box" style="border-left:4px solid #10b981;">
            <div class="reading-header">Support Zone Liquidity Floor</div>
            <div class="reading-body">{reading['support_zone_analysis']}</div>
        </div>
        <div class="reading-box" style="border-left:4px solid #ef4444;">
            <div class="reading-header">Resistance Supply Ceiling</div>
            <div class="reading-body">{reading['resistance_zone_analysis']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        st.markdown(f"""
        <div class="reading-box" style="border-left:4px solid #a855f7;">
            <div class="reading-header">Active Structural Pattern</div>
            <div class="reading-body">{reading['structural_pattern_analysis']}</div>
        </div>
        <div class="reading-box" style="border-left:4px solid #fbbf24;">
            <div class="reading-header">Momentum & Indicator Divergence</div>
            <div class="reading-body">{reading['momentum_and_divergence_analysis']}</div>
        </div>
        <div class="reading-box" style="border-left:4px solid #ec4899;">
            <div class="reading-header">Institutional Volume Activity</div>
            <div class="reading-body">{reading['volume_confirmation_analysis']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### 🎯 **Execution Triggers & Invalidation Framework**")
    tr1, tr2, tr3 = st.columns(3)
    with tr1:
        st.success(f"**Bullish Catalyst Trigger:**\n\n{reading['potential_bullish_trigger']}")
    with tr2:
        st.error(f"**Bearish Risk Trigger:**\n\n{reading['potential_bearish_trigger']}")
    with tr3:
        st.warning(f"**Forecast Invalidation:**\n\n{'; '.join(reading['forecast_invalidation_conditions'])}")


# -----------------------------------------------------------------------------
# TAB 3: TRENDLINES & CHANNELS
# -----------------------------------------------------------------------------
with tab_trend:
    st.markdown("#### **Geometric Trendline & Parallel Channel Engine**")
    
    t_lines = trendlines.get("trendlines", [])
    if t_lines:
        t_rows = []
        for tl in t_lines:
            t_rows.append({
                "Line Type": tl["type"],
                "Start Date": tl["start_date"],
                "Start Price": f"{currency}{tl['start_price']}",
                "End Date": tl["end_date"],
                "Current Level": f"{currency}{tl['end_price']}",
                "Projected 5D Level": f"{currency}{tl['projected_5d_price']}",
                "Verified Touches": tl["touches"],
                "Violations": tl["violations"],
                "Confidence": tl["confidence"]
            })
        st.dataframe(pd.DataFrame(t_rows), use_container_width=True)
    else:
        st.info("No statistically consistent multi-touch trendlines detected in current range.")

    channel = trendlines.get("channel")
    if channel:
        st.markdown("##### 📊 **Parallel Channel Structure**")
        st.info(f"**{channel['type']}**: {channel['interpretation']} (Channel Height: {channel['channel_height_pct']}%)")

    breakouts = trendlines.get("breakouts", [])
    if breakouts:
        st.markdown("##### 🚨 **Detected Breakout / Breakdown Events**")
        for bo in breakouts:
            b_color = "#10b981" if "Breakout" in bo["type"] else "#ef4444"
            st.markdown(f"- **{bo['status']}** ({bo['type']}): {bo['interpretation']}")


# -----------------------------------------------------------------------------
# TAB 4: CLASSICAL CHART PATTERNS
# -----------------------------------------------------------------------------
with tab_patterns:
    st.markdown("#### **Classical Multi-Bar Chart Pattern Geometry**")
    if chart_patterns:
        for cp in chart_patterns:
            p_color = "#10b981" if "Bullish" in cp["bias"] else ("#ef4444" if "Bearish" in cp["bias"] else "#f59e0b")
            st.markdown(f"""
            <div style="background:#111827; border:1px solid #1f293d; border-radius:8px; padding:16px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.15rem; font-weight:700; color:#fff;">🏛️ {cp['pattern']}</span>
                    <span style="color:{p_color}; font-weight:600; font-size:0.85rem; background:rgba(255,255,255,0.06); padding:4px 10px; border-radius:4px;">
                        {cp['bias']} • Confidence: {cp['confidence_score']}
                    </span>
                </div>
                <div style="font-size:0.85rem; color:#94a3b8; margin:6px 0;">
                    Duration: <b>{cp['start_date']}</b> to <b>{cp['end_date']}</b>
                </div>
                <div style="font-size:0.92rem; color:#e2e8f0; margin:8px 0;">
                    <b>Confirmation Criteria:</b> {cp['confirmation_condition']}<br>
                    <b>Invalidation Rule:</b> {cp['invalidation_condition']}
                </div>
                <div style="display:flex; gap:1.5rem; margin-top:8px; font-family:'JetBrains Mono'; font-size:0.88rem;">
                    {f"<span>Measured-Move Target: <b style='color:#38bdf8;'>{currency}{cp['measured_move_target']}</b></span>" if 'measured_move_target' in cp else ""}
                    {f"<span>Neckline / Support: <b>{currency}{cp['neckline_level']}</b></span>" if 'neckline_level' in cp else ""}
                </div>
                <div style="font-size:0.78rem; color:#64748b; margin-top:6px; font-style:italic;">
                    Empirical validation: {cp.get('historical_hit_rate', 'Evaluated against out-of-sample data')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No mature classical chart pattern formations (Double Tops/Bottoms, Triangles, Flags, Wedges) active at this moment.")


# -----------------------------------------------------------------------------
# TAB 5: CANDLESTICK PATTERNS
# -----------------------------------------------------------------------------
with tab_candlestick:
    st.markdown("#### **Candlestick Pattern Recognition Engine**")
    st.caption("Identifies single, double, and triple candle anatomy with prior trend context. Candlestick signals are contextual indicators and may fail during choppy or illiquid periods.")
    
    if candlestick_patterns:
        c_rows = []
        for p in reversed(candlestick_patterns):
            c_rows.append({
                "Date": p["date"],
                "Pattern Name": p["pattern"],
                "Directional Bias": p["bias"],
                "Price": f"{currency}{p['price']}",
                "Significance": p["significance"],
                "Failure Risk": p["failure_risk"]
            })
        st.dataframe(pd.DataFrame(c_rows), use_container_width=True)
    else:
        st.info("No standard single or multi-candle reversal patterns detected in the recent lookback.")


# -----------------------------------------------------------------------------
# TAB 6: 5-DAY FORECAST SCENARIOS
# -----------------------------------------------------------------------------
with tab_scenarios:
    st.markdown("#### **Forward 5-Trading-Session Price Scenarios (P90 / P50 / P10)**")
    st.caption(forecast_scenarios["disclaimer"])

    fut_dates = forecast_scenarios.get("future_dates", [])
    bull_scen = forecast_scenarios["bullish_scenario"]
    base_scen = forecast_scenarios["base_scenario"]
    bear_scen = forecast_scenarios["bearish_scenario"]

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(f"""
        <div class="metric-card" style="border-left:4px solid #10b981;">
            <div class="metric-label">{bull_scen['name']}</div>
            <div class="metric-value" style="color:#10b981;">+{bull_scen['target_return_pct']}%</div>
            <div class="metric-delta">Target: {currency}{bull_scen['target_price']} on {fut_dates[-1] if fut_dates else 'Day 5'}</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:8px;">
                <b>Strengthens:</b> {bull_scen['strengthen_condition']}<br>
                <b>Invalidates:</b> {bull_scen['invalidation_condition']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="metric-card" style="border-left:4px solid #f59e0b;">
            <div class="metric-label">{base_scen['name']}</div>
            <div class="metric-value">{'+' if base_scen['target_return_pct']>=0 else ''}{base_scen['target_return_pct']}%</div>
            <div class="metric-delta">Target: {currency}{base_scen['target_price']} on {fut_dates[-1] if fut_dates else 'Day 5'}</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:8px;">
                <b>Strengthens:</b> {base_scen['strengthen_condition']}<br>
                <b>Invalidates:</b> {base_scen['invalidation_condition']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div class="metric-card" style="border-left:4px solid #ef4444;">
            <div class="metric-label">{bear_scen['name']}</div>
            <div class="metric-value" style="color:#ef4444;">{bear_scen['target_return_pct']}%</div>
            <div class="metric-delta">Target: {currency}{bear_scen['target_price']} on {fut_dates[-1] if fut_dates else 'Day 5'}</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:8px;">
                <b>Strengthens:</b> {bear_scen['strengthen_condition']}<br>
                <b>Invalidates:</b> {bear_scen['invalidation_condition']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Future dates table
    traj_df = pd.DataFrame({
        "Trading Day": [f"Day {i+1} ({d})" for i, d in enumerate(fut_dates)],
        "Bullish Path (P90)": [f"{currency}{p}" for p in bull_scen["trajectory"]],
        "Base Equilibrium (P50)": [f"{currency}{p}" for p in base_scen["trajectory"]],
        "Bearish Path (P10)": [f"{currency}{p}" for p in bear_scen["trajectory"]],
        "Upper Cone": [f"{currency}{p}" for p in forecast_scenarios["uncertainty_cone"]["upper_band"]],
        "Lower Cone": [f"{currency}{p}" for p in forecast_scenarios["uncertainty_cone"]["lower_band"]]
    })
    st.table(traj_df)


# -----------------------------------------------------------------------------
# TAB 7: SUPPORT & RESISTANCE ENGINE
# -----------------------------------------------------------------------------
with tab_sr:
    st.markdown("#### **Dedicated Support and Resistance Intelligence Engine**")
    br = sr.get("breakdown_risk_summary", {})
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Primary Support Zone", br.get("primary_support_zone", "N/A"))
    with col_s2:
        st.metric("Buffer to Support", f"{br.get('distance_to_primary_support_pct', 'N/A')}%")
    with col_s3:
        st.metric("Breakdown Freq", f"{br.get('historical_breakdown_frequency_pct', 'N/A')}%")
    with col_s4:
        st.metric("Breakdown Risk Rating", br.get("risk_rating", "N/A"))

    st.markdown("##### 🟢 **Identified Support Zones**")
    s_zones = sr.get("support_zones", [])
    if s_zones:
        s_rows = []
        for z in s_zones:
            s_rows.append({
                "Zone Range": f"{currency}{z['lower_boundary']} – {currency}{z['upper_boundary']}",
                "Confirmed Tests": z["meaningful_tests"],
                "Avg Rebound": f"+{z['average_reaction_pct']}%",
                "Last Interaction": f"{z['recent_interaction_date']} ({z['days_since_interaction']}d ago)",
                "Distance Cushion": z["distance_description"],
                "Volume Signature": z["volume_evidence"],
                "Evidence Strength": z["confidence_label"]
            })
        st.dataframe(pd.DataFrame(s_rows), use_container_width=True)

    st.markdown("##### 🔴 **Identified Resistance Zones**")
    r_zones = sr.get("resistance_zones", [])
    if r_zones:
        r_rows = []
        for z in r_zones:
            r_rows.append({
                "Zone Range": f"{currency}{z['lower_boundary']} – {currency}{z['upper_boundary']}",
                "Confirmed Tests": z["meaningful_tests"],
                "Avg Rejection": f"-{z['average_reaction_pct']}%",
                "Last Interaction": f"{z['recent_interaction_date']} ({z['days_since_interaction']}d ago)",
                "Distance To Resistance": z["distance_description"],
                "Volume Signature": z["volume_evidence"],
                "Evidence Strength": z["confidence_label"]
            })
        st.dataframe(pd.DataFrame(r_rows), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 8: 11 CORE QUESTIONS
# -----------------------------------------------------------------------------
with tab_qa:
    st.markdown("#### **Executive Answers to the 11 Core Market Questions**")
    for key, item in qa.items():
        st.markdown(f"""
        <div class="qa-box">
            <div class="qa-q">{item['question']}</div>
            <div class="qa-a">{item['answer']}</div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 9: SIGNAL BACKTESTING MATRIX
# -----------------------------------------------------------------------------
with tab_backtest:
    st.markdown("#### **Historical Signal Validation & Empirical Hit Rates**")
    st.caption(signal_backtest.get("methodology", ""))
    
    sigs = signal_backtest.get("signal_summaries", [])
    if sigs:
        sig_df = pd.DataFrame(sigs)
        sig_df.columns = ["Category", "Signal Name", "Sample Size", "5-Day Hit Rate %", "Avg Subsequent 5D Move %", "Significance"]
        st.dataframe(sig_df, use_container_width=True)
    else:
        st.info("Insufficient historical pattern instances to form a statistically valid backtest sample.")

    st.markdown("---")
    st.markdown("##### 🧪 **Walk-Forward ML Directional Validation**")
    if val.get("status") == "SUCCESS":
        st.write(f"Production Model Accuracy: **{val['accuracy']['production_model_accuracy_pct']}%** vs Baseline: **{val['accuracy']['simple_baseline_accuracy_pct']}%**")
        st.write(f"Bearish Warning Quality: **Precision {val['bearish_alert_quality']['precision_pct']}%**, **Recall {val['bearish_alert_quality']['recall_pct']}%**")
        st.caption(val["scientific_integrity_statement"])


# -----------------------------------------------------------------------------
# TAB 10: NEWS INTELLIGENCE
# -----------------------------------------------------------------------------
with tab_news:
    st.markdown("#### **News Intelligence & Financial Sentiment**")
    st.write(f"Aggregate News Flow: **{news.get('aggregate_sentiment')}** (Score: {news.get('aggregate_score')})")
    
    for art in news.get("articles", []):
        s_color = "#10b981" if art["sentiment_label"] == "Positive" else ("#ef4444" if art["sentiment_label"] == "Negative" else "#f59e0b")
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1f293d; border-radius:6px; padding:12px 16px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.75rem; color:#94a3b8; font-family:'JetBrains Mono';">{art['publisher']} • {art['publication_time']}</span>
                <span style="font-size:0.75rem; background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:4px; color:{s_color}; font-weight:600;">
                    {art['event_category']} | {art['sentiment_label']} ({art['sentiment_score']})
                </span>
            </div>
            <div style="font-weight:600; font-size:0.95rem; margin:6px 0;">
                <a href="{art['link']}" target="_blank" style="color:#60a5fa; text-decoration:none;">{art['headline']} ↗</a>
            </div>
            <div style="font-size:0.83rem; color:#cbd5e1; font-style:italic;">
                💡 Impact: {art['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
