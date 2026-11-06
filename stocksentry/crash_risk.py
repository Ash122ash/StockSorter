"""
StockSentry AI - Crash-Risk Intelligence Engine
Independent downside tail-risk assessment module distinguishing ordinary bearish fluctuations
from asymmetric capital destruction events (>= 5% collapse in 5 sessions).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from stocksentry.config import CRASH_RISK_CONFIG


class CrashRiskEngine:
    """
    Dedicated crash & tail-risk diagnostics engine.
    Analyzes volatility spikes, gap downs, ATR expansion, support distance, and drawdown history.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or CRASH_RISK_CONFIG
        self.crash_thresh_pct = self.cfg.get("crash_threshold_pct", -5.0)

    def assess_risk(
        self,
        df: pd.DataFrame,
        symbol: str,
        support_zones: Optional[List[Dict[str, Any]]] = None,
        news_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates multi-factor tail-risk score and empirical crash probability.
        """
        if df is None or len(df) < 30:
            return {
                "risk_category": "INDETERMINATE",
                "risk_score_100": 0,
                "message": "Insufficient historical data for tail-risk modeling."
            }

        d = df.copy()
        close = d["Close"]
        high = d["High"]
        low = d["Low"]
        volume = d["Volume"]

        # 1. Historical Maximum Drawdown
        running_max = close.cummax()
        drawdown_series = ((close - running_max) / running_max) * 100.0
        max_drawdown = float(drawdown_series.min())
        current_drawdown = float(drawdown_series.iloc[-1])

        # 2. Historical annualized volatility & recent spike
        log_ret = np.log(close / close.shift(1))
        ann_vol = float(log_ret.rolling(20, min_periods=5).std().iloc[-1] * np.sqrt(252) * 100.0)
        long_term_vol = float(log_ret.std() * np.sqrt(252) * 100.0)
        vol_spike_ratio = (ann_vol / long_term_vol) if long_term_vol > 0 else 1.0

        # 3. ATR Expansion
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_fast = float(tr.rolling(5, min_periods=3).mean().iloc[-1])
        atr_slow = float(tr.rolling(20, min_periods=10).mean().iloc[-1])
        atr_expansion_ratio = (atr_fast / atr_slow) if atr_slow > 0 else 1.0

        # 4. Opening gaps down in last 5 sessions
        recent_gaps = []
        for i in range(-5, 0):
            if abs(i) < len(d):
                gap = ((d["Open"].iloc[i] / d["Close"].iloc[i-1]) - 1.0) * 100.0
                if gap < -1.0:
                    recent_gaps.append(gap)

        # 5. Distance to closest major support zone
        current_price = float(close.iloc[-1])
        support_distance_pct = 999.0
        primary_support_info = "No major support zone identified nearby"
        is_support_broken = False

        if support_zones:
            for sz in support_zones:
                if sz.get("is_broken", False):
                    is_support_broken = True
                dist = sz.get("distance_pct", 999.0)
                if 0 <= dist < support_distance_pct:
                    support_distance_pct = dist
                    primary_support_info = f"{sz['lower_boundary']} – {sz['upper_boundary']} ({dist:.1f}% below current price)"

        # 6. Empirical Probability of a material crash event (return <= crash_threshold_pct in 5 sessions)
        fwd_5d_ret = ((close.shift(-5) / close) - 1.0) * 100.0
        valid_fwd = fwd_5d_ret.dropna()
        sample_size = len(valid_fwd)

        crash_prob_pct = None
        if sample_size >= 60:
            crashes_count = int((valid_fwd <= self.crash_thresh_pct).sum())
            crash_prob_pct = round((crashes_count / sample_size) * 100.0, 1)

        # 7. Multi-factor Risk Scoring (0 to 100)
        risk_score = 15.0 # baseline market entropy

        risk_factors = []
        mitigating_factors = []

        if vol_spike_ratio > 1.3:
            risk_score += 20.0
            risk_factors.append(f"Abnormal 20-day volatility expansion ({ann_vol:.1f}% vs baseline {long_term_vol:.1f}%).")
        else:
            mitigating_factors.append("Current volatility is within normal historical boundaries.")

        if atr_expansion_ratio > 1.25:
            risk_score += 15.0
            risk_factors.append(f"Short-term ATR expanding rapidly ({atr_expansion_ratio:.2f}x of 20-day average), signaling selling turbulence.")

        if len(recent_gaps) > 0:
            risk_score += 15.0
            risk_factors.append(f"{len(recent_gaps)} significant gap-down opening(s) observed over the last 5 trading days.")

        if is_support_broken:
            risk_score += 20.0
            risk_factors.append("Recent technical support zone has been breached, removing historical liquidity cushioning.")
        elif support_distance_pct < 2.5:
            risk_score += 10.0
            risk_factors.append("Price is testing immediate support floor; failure to hold triggers stop-loss cascades.")
        else:
            mitigating_factors.append(f"Adequate support buffer ({support_distance_pct:.1f}% cushion to nearest demand zone).")

        if current_drawdown < -20.0:
            risk_score += 10.0
            risk_factors.append(f"Asset is currently in deep corrective territory ({current_drawdown:.1f}% below peak).")
        elif current_drawdown > -5.0:
            mitigating_factors.append("Trading near cycle highs without structural drawdown damage.")

        if news_score <= -0.3:
            risk_score += 15.0
            risk_factors.append("Adverse corporate headline flow / regulatory headwind elevating tail-risk vulnerability.")
        elif news_score >= 0.2:
            mitigating_factors.append("Favorable news sentiment counteracts downside contagion.")

        # Cap score
        risk_score = min(100.0, max(5.0, risk_score))

        # Categorization
        if risk_score >= 70.0:
            risk_category = "SEVERE"
            risk_color = "#ef4444" # red
        elif risk_score >= 50.0:
            risk_category = "HIGH"
            risk_color = "#f97316" # orange
        elif risk_score >= 30.0:
            risk_category = "ELEVATED"
            risk_color = "#eab308" # yellow
        else:
            risk_category = "LOW"
            risk_color = "#22c55e" # green

        downside_scenario_pct = round(self.crash_thresh_pct * (1.0 + (risk_score / 100.0)), 2)
        downside_price = round(current_price * (1.0 + (downside_scenario_pct / 100.0)), 2)

        return {
            "symbol": symbol,
            "risk_category": risk_category,
            "risk_color": risk_color,
            "risk_score_100": round(risk_score, 1),
            "crash_definition": f"Return <= {self.crash_thresh_pct}% over a 5-trading-session window",
            "historical_crash_probability_pct": crash_prob_pct,
            "historical_sample_size": sample_size,
            "historical_max_drawdown_pct": round(max_drawdown, 2),
            "current_drawdown_from_peak_pct": round(current_drawdown, 2),
            "recent_annualized_volatility_pct": round(ann_vol, 2),
            "primary_support_zone": primary_support_info,
            "downside_stress_scenario": {
                "percentage_fall": downside_scenario_pct,
                "projected_price": downside_price
            },
            "key_risk_drivers": risk_factors if risk_factors else ["No acute systemic or idiosyncratic warning flags."],
            "mitigating_factors": mitigating_factors,
            "risk_escalators": [
                "Index gap-down below 50-day moving average",
                "High-volume break below primary support zone",
                "Negative regulatory audit or forensic disclosure"
            ],
            "risk_reducers": [
                "Consolidation inside support band with shrinking ATR",
                "Bullish divergence on RSI or institutional block accumulation",
                "Positive earnings beat with forward margin expansion"
            ]
        }


# Global default crash risk engine
crash_engine = CrashRiskEngine()
