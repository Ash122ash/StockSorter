"""
NSE Stock Sorter — backend
---------------------------
Fetches the list of stocks trading on the NSE (India), pulls their latest
price via yfinance, caches the result, and serves it as JSON to the
frontend table (which handles the ascending/descending sorting client-side).

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import io
import json
import time
import threading
from pathlib import Path

# Remove invalid CA bundle environment variables if they exist
os.environ.pop('REQUESTS_CA_BUNDLE', None)
os.environ.pop('CURL_CA_BUNDLE', None)

import requests
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

SEED_DATA_DIR = Path(__file__).parent / "data"
_data_env = os.environ.get("DATA_DIR")
if _data_env:
    DATA_DIR = Path(_data_env)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SEED_DATA_DIR.exists():
        import shutil
        for item in SEED_DATA_DIR.glob("*.json"):
            dest = DATA_DIR / item.name
            if not dest.exists():
                try:
                    shutil.copy2(item, dest)
                except Exception:
                    pass
else:
    DATA_DIR = SEED_DATA_DIR
    DATA_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

# NSE's official master list of every equity symbol currently listed.
NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_ETF_LIST_URL = "https://archives.nseindia.com/content/equities/eq_etfseclist.csv"
BSE_EQUITY_LIST_URL = "https://raw.githubusercontent.com/RuchiTanmay/bseindia/master/bseindia/bse_security_list.csv"

# Hand-maintained fallback / smaller universes so the app is usable even
# when the full ~2000-symbol download would be slow, and as a fallback if
# the NSE archive endpoint is ever unreachable. Edit freely to keep current.
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL",
    "ITC", "SBIN", "LICI", "HINDUNILVR", "LT", "BAJFINANCE", "HCLTECH",
    "MARUTI", "SUNPHARMA", "KOTAKBANK", "AXISBANK", "M&M", "ULTRACEMCO",
    "NTPC", "TITAN", "ONGC", "ADANIENT", "TATAMOTORS", "ADANIPORTS",
    "ASIANPAINT", "BAJAJFINSV", "POWERGRID", "COALINDIA", "WIPRO",
    "NESTLEIND", "JSWSTEEL", "TATASTEEL", "BAJAJ-AUTO", "GRASIM",
    "TECHM", "HINDALCO", "SBILIFE", "HDFCLIFE", "CIPLA", "DRREDDY",
    "APOLLOHOSP", "EICHERMOT", "BRITANNIA", "DIVISLAB", "INDUSINDBK",
    "HEROMOTOCO", "BPCL", "TATACONSUM", "SHRIRAMFIN"
]

UNIVERSES = {
    "combined": None,
    "nifty50": NIFTY_50,
    "all": None,       # resolved dynamically from the NSE archive CSV
    "nse_etf": None,   # resolved dynamically from the NSE ETF CSV
    "bse_all": None,   # resolved dynamically from the BSE CSV
}


def _cache_path(universe: str) -> Path:
    return DATA_DIR / f"stocks_{universe}.json"


def fetch_all_nse_symbols() -> list[dict]:
    """Download the full list of NSE-listed equities (symbol + company name)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockSorter/1.0)"}
    resp = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    records = [
        {"symbol": row["SYMBOL"].strip(), "name": row["NAME OF COMPANY"].strip()}
        for _, row in df.iterrows()
    ]
    return records


def fetch_nse_etf_symbols() -> list[dict]:
    """Download the list of NSE Exchange Traded Funds (ETFs) (symbol + name)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockSorter/1.0)"}
    resp = requests.get(NSE_ETF_LIST_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    records = [
        {"symbol": row["Symbol"].strip(), "name": row["SecurityName"].strip()}
        for _, row in df.iterrows()
    ]
    return sorted(records, key=lambda x: x["symbol"])


def fetch_bse_symbols() -> list[dict]:
    """Download the list of BSE equities (symbol + name)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockSorter/1.0)"}
    resp = requests.get(BSE_EQUITY_LIST_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.dropna(subset=["symbol", "security_name"])
    df = df.drop_duplicates(subset=["symbol"])
    records = [
        {"symbol": str(row["symbol"]).strip(), "name": str(row["security_name"]).strip()}
        for _, row in df.iterrows()
    ]
    return sorted(records, key=lambda x: x["symbol"])


def fetch_prices(symbols: list[str], suffix: str = ".NS") -> dict:
    """Batch-fetch latest price + prior close for a list of symbols via yfinance."""
    tickers = [f"{s}{suffix}" for s in symbols]
    prices: dict[str, dict] = {}

    chunk_size = 150
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(
                tickers=chunk,
                period="2d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=False,
            )
        except Exception:
            continue

        for t in chunk:
            symbol = t.replace(suffix, "")
            try:
                if len(chunk) == 1:
                    closes = data["Close"].dropna()
                else:
                    closes = data[t]["Close"].dropna()
                if closes.empty:
                    continue
                last_price = float(closes.iloc[-1])
                prev_price = float(closes.iloc[-2]) if len(closes) > 1 else last_price
                change = last_price - prev_price
                change_pct = (change / prev_price * 100) if prev_price else 0.0
                prices[symbol] = {
                    "price": round(last_price, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                }
            except Exception:
                continue

    return prices


def build_universe(universe: str) -> list[dict]:
    if universe == "all":
        base = fetch_all_nse_symbols()
    elif universe == "nse_etf":
        base = fetch_nse_etf_symbols()
    elif universe == "bse_all":
        base = fetch_bse_symbols()
    else:
        symbols = UNIVERSES.get(universe, NIFTY_50)
        base = [{"symbol": s, "name": s} for s in symbols]
    return base


_refreshing_universes = set()
_refresh_lock = threading.Lock()


def _do_actual_refresh(universe: str) -> dict:
    """Internal function to fetch prices and update the cache file."""
    path = _cache_path(universe)
    try:
        if universe == "combined":
            # Refresh components first
            nse_all = _do_actual_refresh("all")
            nse_etf = _do_actual_refresh("nse_etf")
            bse_all = _do_actual_refresh("bse_all")

            merged_stocks = []
            for s in nse_all.get("stocks", []):
                s_copy = dict(s)
                s_copy["exchange"] = "NSE"
                merged_stocks.append(s_copy)
            for s in nse_etf.get("stocks", []):
                s_copy = dict(s)
                s_copy["exchange"] = "NSE ETF"
                merged_stocks.append(s_copy)
            for s in bse_all.get("stocks", []):
                s_copy = dict(s)
                s_copy["exchange"] = "BSE"
                merged_stocks.append(s_copy)

            merged_stocks.sort(key=lambda x: x["symbol"])
            result = {
                "fetched_at": time.time(),
                "count": len(merged_stocks),
                "stocks": merged_stocks
            }
            path.write_text(json.dumps(result), encoding="utf-8")
            return result

        base = build_universe(universe)
        symbols = [b["symbol"] for b in base]
        suffix = ".BO" if universe == "bse_all" else ".NS"
        prices = fetch_prices(symbols, suffix=suffix)

        exchange = "NSE"
        if universe == "nse_etf":
            exchange = "NSE ETF"
        elif universe == "bse_all":
            exchange = "BSE"

        rows = []
        for b in base:
            p = prices.get(b["symbol"])
            if p is None:
                continue
            rows.append({
                "symbol": b["symbol"],
                "name": b["name"],
                "exchange": exchange,
                "price": p["price"],
                "change": p["change"],
                "change_pct": p["change_pct"],
            })

        result = {"fetched_at": time.time(), "count": len(rows), "stocks": rows}
        path.write_text(json.dumps(result), encoding="utf-8")
        return result
    except Exception as e:
        print(f"Error refreshing {universe}: {e}")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return {"fetched_at": time.time(), "count": 0, "stocks": []}


def _trigger_background_refresh(universe: str):
    with _refresh_lock:
        if universe in _refreshing_universes:
            return
        _refreshing_universes.add(universe)

    def _worker():
        try:
            _do_actual_refresh(universe)
        finally:
            with _refresh_lock:
                _refreshing_universes.discard(universe)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def load_or_refresh(universe: str, force: bool = False) -> dict:
    """
    Instantaneous response using Stale-While-Revalidate:
    - If cache exists on disk, returns immediately (< 50ms).
    - If stale or forced, updates in a background thread so the user is never blocked.
    """
    path = _cache_path(universe)

    # 1. Check if cache exists
    if path.exists():
        try:
            cached_data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if cached_data and cached_data.get("stocks"):
                age = time.time() - path.stat().st_mtime
                if force or age >= CACHE_TTL_SECONDS:
                    _trigger_background_refresh(universe)
                return cached_data
        except Exception:
            pass

    # 2. If combined does not have cache, synthesize from components immediately
    if universe == "combined":
        nse_all = load_or_refresh("all", force=False)
        nse_etf = load_or_refresh("nse_etf", force=False)
        bse_all = load_or_refresh("bse_all", force=False)

        merged_stocks = []
        for s in nse_all.get("stocks", []):
            s_copy = dict(s)
            s_copy["exchange"] = "NSE"
            merged_stocks.append(s_copy)
        for s in nse_etf.get("stocks", []):
            s_copy = dict(s)
            s_copy["exchange"] = "NSE ETF"
            merged_stocks.append(s_copy)
        for s in bse_all.get("stocks", []):
            s_copy = dict(s)
            s_copy["exchange"] = "BSE"
            merged_stocks.append(s_copy)

        merged_stocks.sort(key=lambda x: x["symbol"])
        result = {
            "fetched_at": time.time(),
            "count": len(merged_stocks),
            "stocks": merged_stocks
        }
        try:
            path.write_text(json.dumps(result), encoding="utf-8")
        except Exception:
            pass
        return result

    # 3. Synchronous fallback if no cache exists at all
    return _do_actual_refresh(universe)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/advanced")
def advanced():
    return render_template("advanced.html")


@app.route("/api/stocks")
def api_stocks():
    universe = request.args.get("universe", "combined")
    force = request.args.get("refresh", "0") == "1"
    if universe not in UNIVERSES:
        universe = "combined"

    try:
        result = load_or_refresh(universe, force=force)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/stocksentry/analyze")
def api_stocksentry_analyze():
    symbol = request.args.get("symbol", "RELIANCE.NS").strip().upper()
    period = request.args.get("period", "2y")
    force = request.args.get("refresh", "0") == "1"
    try:
        from stocksentry.orchestrator import run_full_analysis
        data = run_full_analysis(symbol, period=period, force_refresh=force)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/stocksentry/launch_streamlit", methods=["POST", "GET"])
def api_launch_streamlit():
    try:
        import socket
        import subprocess
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        res = sock.connect_ex(('127.0.0.1', 8501))
        sock.close()
        if res != 0:
            script_path = Path(__file__).parent / "streamlit_app.py"
            py_exe = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
            cmd = [str(py_exe), "-m", "streamlit", "run", str(script_path), "--server.headless", "true", "--server.port", "8501"]
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0)
            time.sleep(2.0)
        return jsonify({"ok": True, "url": "http://localhost:8501"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
