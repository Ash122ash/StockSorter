# NSE Stock Sorter & StockSentry AI Platform

A comprehensive equity analysis environment comprising:
1. **NSE Stock Sorter**: Fast universe explorer (Combined, Nifty 50, All NSE, NSE ETF, BSE) with real-time client-side sorting and filtering.
2. **StockSentry AI**: An institutional-grade quantitative stock analysis & crash-risk intelligence terminal with automated technical chart pattern recognition, trendline geometry, and forward scenario projections.

---

## 🛡️ StockSentry AI & Advanced Chart Analysis Workstation

StockSentry AI is accessed directly by clicking the **"🛡️ StockSentry AI"** button on the top right of the main page (opening `/advanced` in a new window/tab), or running the standalone Streamlit dashboard.

### 📈 Advanced Chart Analysis Capabilities
1. **Interactive Candlestick Chart (Plotly)**: 3-pane synchronized workstation with OHLC candles, EMA 9/21, SMA 20/50, Bollinger Bands with shaded fill, horizontal S/R liquidity bands, fitted trendlines, candlestick pattern markers, volume bars, and RSI 14 with divergence annotations.
2. **Automated Support & Resistance Drawing**: Swing extrema clustering with ATR buffer, multi-touch verification without double-counting, recency decay weighting, breach detection, and empirical breakdown risk.
3. **Trendlines & Channels**: Algorithmic multi-touch line fitting, ascending/descending channel detection, and breakout confirmation with volume validation (RVOL).
4. **Classical Chart Pattern Recognition**: Algorithmic detection of Double Tops/Bottoms, Head & Shoulders, Inverse H&S, Triangles (Ascending, Descending, Symmetrical), Bull/Bear Flags, and Wedges with technical measured-move targets.
5. **Candlestick Pattern Recognition**: Geometric candle anatomy scanner for Doji, Hammer, Inverted Hammer, Shooting Star, Hanging Man, Bullish/Bearish Engulfing, Morning/Evening Star, Piercing Line, and Dark Cloud Cover with preceding trend context and failure risk disclosures.
6. **RSI, MACD & Divergence Engine**: Real-time identification of regular and hidden Bullish/Bearish divergences strictly comparing price swing extrema with indicator extrema, plus EMA 9/21 crossovers.
7. **Technical Analyst's Reading**: Deterministically generated analytical narrative synthesizing trend, structure, liquidity zones, patterns, momentum, volume, catalysts, and invalidation rules.
8. **Forward 5-Session Price Scenarios**: Plotted strictly into future trading days with dashed trajectories (Bullish P90, Base P50, Bearish P10) and shaded uncertainty cones.
9. **Signal Historical Backtesting Matrix**: Chronological out-of-sample evaluation of chart signals, reporting sample sizes, 5-day hit rates, and subsequent return distributions.

---

## 🚀 Quickstart

### 1. Installation
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Web Application
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser:
- Main Table: Browse and sort NSE / BSE / ETF equities.
- Top Right Button: Click **"🛡️ StockSentry AI"** to open the full quantitative analysis terminal at **http://127.0.0.1:5000/advanced**.

### 3. Running the Standalone Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Or click the **"🚀 Open Streamlit Studio"** button directly from inside the `/advanced` terminal topbar.

---

## 🧪 Testing

Run the automated test suite covering all 15 unit and integration tests:

```bash
pytest -v tests/
```
