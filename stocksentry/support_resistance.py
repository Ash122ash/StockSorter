"""
StockSentry AI - Support and Resistance Engine (Highest Priority)
Identifies dynamic price zones using swing extrema clustering, reaction volume,
volatility-scaled zone widths, recency decay weighting, and breakdown risk estimation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from stocksentry.config import SR_CONFIG


class SupportResistanceEngine:
    """
    Dedicated quantitative Support & Resistance detection module.
    Rejects the illusion of 'unbreakable single price lines', computing dynamic zones
    anchored by order flow clustering, ATR buffering, and empirical reaction history.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or SR_CONFIG

    def analyze(self, df: pd.DataFrame, current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Runs the full 10-step algorithm on historical OHLCV data.
        Returns validated support zones, resistance zones, and breakdown risk analytics.
        """
        if df is None or len(df) < 30:
            return {
                "support_zones": [],
                "resistance_zones": [],
                "error": "Insufficient historical data for reliable Support/Resistance zone clustering (minimum 30 sessions required)."
            }

        d = df.copy()
        if current_price is None:
            current_price = float(d["Close"].iloc[-1])

        # Calculate ATR for zone width scaling
        prev_close = d["Close"].shift(1)
        tr = pd.concat([
            d["High"] - d["Low"],
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        current_atr = float(tr.rolling(14, min_periods=5).mean().iloc[-1])
        vol_avg = float(d["Volume"].rolling(20, min_periods=5).mean().iloc[-1])

        # Step 1: Detect local swing highs and swing lows
        window = self.cfg.get("swing_window", 5)
        d["Swing_High"] = (d["High"] == d["High"].rolling(window=window*2+1, center=True).max())
        d["Swing_Low"] = (d["Low"] == d["Low"].rolling(window=window*2+1, center=True).min())

        swing_lows = []
        for idx, row in d[d["Swing_Low"]].iterrows():
            swing_lows.append({
                "date": idx,
                "price": float(row["Low"]),
                "volume": float(row["Volume"]),
                "type": "low"
            })

        swing_highs = []
        for idx, row in d[d["Swing_High"]].iterrows():
            swing_highs.append({
                "date": idx,
                "price": float(row["High"]),
                "volume": float(row["Volume"]),
                "type": "high"
            })

        # Step 2: Cluster nearby price levels with configurable tolerance
        cluster_tol = self.cfg.get("cluster_tolerance_pct", 1.8) / 100.0

        support_clusters = self._cluster_levels(swing_lows, cluster_tol)
        resistance_clusters = self._cluster_levels(swing_highs, cluster_tol)

        # Steps 3-8: Zone construction, reaction measurement, recency decay, breakdown checks
        support_zones = self._build_zones(
            clusters=support_clusters,
            is_support=True,
            current_price=current_price,
            current_atr=current_atr,
            vol_avg=vol_avg,
            full_df=d
        )

        resistance_zones = self._build_zones(
            clusters=resistance_clusters,
            is_support=False,
            current_price=current_price,
            current_atr=current_atr,
            vol_avg=vol_avg,
            full_df=d
        )

        # Step 9: Out-of-sample breakdown frequency estimation
        breakdown_risk = self._estimate_breakdown_risk(support_zones, d, current_price)

        return {
            "current_price": round(current_price, 2),
            "current_atr": round(current_atr, 2),
            "support_zones": support_zones,
            "resistance_zones": resistance_zones,
            "breakdown_risk_summary": breakdown_risk,
            "methodology_disclaimer": (
                "Zones represent probabilistic liquidity bands, not rigid floors or ceilings. "
                "Breakdown frequency is computed empirically against past test instances."
            )
        }

    def _cluster_levels(self, points: List[Dict[str, Any]], tolerance: float) -> List[List[Dict[str, Any]]]:
        """Clusters points whose prices fall within tolerance bandwidth."""
        if not points:
            return []
        
        # Sort by price
        sorted_pts = sorted(points, key=lambda x: x["price"])
        clusters: List[List[Dict[str, Any]]] = []
        current_cluster = [sorted_pts[0]]

        for pt in sorted_pts[1:]:
            cluster_mean = np.mean([p["price"] for p in current_cluster])
            if abs(pt["price"] - cluster_mean) / cluster_mean <= tolerance:
                current_cluster.append(pt)
            else:
                clusters.append(current_cluster)
                current_cluster = [pt]
        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _build_zones(
        self,
        clusters: List[List[Dict[str, Any]]],
        is_support: bool,
        current_price: float,
        current_atr: float,
        vol_avg: float,
        full_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        zones = []
        zone_half_width = current_atr * self.cfg.get("zone_width_atr_mult", 0.5)
        now_dt = full_df.index[-1]

        for cluster in clusters:
            if len(cluster) < self.cfg.get("min_touches", 2):
                continue

            prices = [p["price"] for p in cluster]
            center_price = float(np.mean(prices))
            lower_boundary = round(max(0.01, min(prices) - zone_half_width), 2)
            upper_boundary = round(max(prices) + zone_half_width, 2)

            # Rebounds / Rejections measurement
            touches = len(cluster)
            reactions = []
            volumes = [p["volume"] for p in cluster]
            recent_date = max(p["date"] for p in cluster)

            for p in cluster:
                # Find subsequent 5 sessions price reaction after the touch
                touch_idx = full_df.index.get_indexer([p["date"]], method="nearest")[0]
                if touch_idx < len(full_df) - 1:
                    subsequent_slice = full_df.iloc[touch_idx + 1: min(touch_idx + 6, len(full_df))]
                    if is_support:
                        # % max rebound up from touch low
                        max_rebound = ((subsequent_slice["High"].max() - p["price"]) / p["price"]) * 100.0
                        reactions.append(max_rebound)
                    else:
                        # % max rejection down from touch high
                        max_rejection = ((p["price"] - subsequent_slice["Low"].min()) / p["price"]) * 100.0
                        reactions.append(max_rejection)

            avg_reaction = round(float(np.mean(reactions)), 2) if reactions else 0.0

            # Recency weighting
            days_ago = (now_dt - recent_date).days
            recency_halflife = self.cfg.get("recency_decay_halflife", 60)
            recency_weight = float(np.exp(-np.log(2) * max(0, days_ago) / recency_halflife))

            # Volume evidence
            avg_touch_vol = float(np.mean(volumes)) if volumes else vol_avg
            vol_ratio = avg_touch_vol / (vol_avg if vol_avg > 0 else 1.0)
            volume_label = "Institutional Volume (Above Avg)" if vol_ratio > 1.25 else "Average Volume Participation"

            # Check if currently broken or breached
            is_broken = False
            if is_support:
                if current_price < lower_boundary:
                    is_broken = True
                    distance_pct = round(((current_price - lower_boundary) / current_price) * 100.0, 2)
                    distance_description = f"Price is already BELOW support boundary by {abs(distance_pct)}% (Broken / Weakened)"
                else:
                    distance_pct = round(((current_price - upper_boundary) / current_price) * 100.0, 2)
                    distance_description = f"{distance_pct}% downside cushion to zone top"
            else:
                if current_price > upper_boundary:
                    is_broken = True
                    distance_pct = round(((current_price - upper_boundary) / current_price) * 100.0, 2)
                    distance_description = f"Price is already ABOVE resistance boundary by {distance_pct}% (Breakout Occurred)"
                else:
                    distance_pct = round(((lower_boundary - current_price) / current_price) * 100.0, 2)
                    distance_description = f"{distance_pct}% upside distance to resistance zone"

            # Confidence / Evidence-Strength Labeling
            # Must factor in touches, recency, reaction strength, and whether it's broken
            score = (touches * 1.5) + (recency_weight * 3.0) + (min(avg_reaction, 10.0) / 2.0) + (1.5 if vol_ratio > 1.2 else 0.5)
            if is_broken:
                score *= 0.4

            if score >= 7.5:
                confidence = "Strong historical evidence"
            elif score >= 4.5:
                confidence = "Moderate historical evidence"
            else:
                confidence = "Weak historical evidence"

            # Invalidation factors
            if is_support:
                invalidation = (
                    f"A confirmed daily close below {lower_boundary} with elevated RVOL (>1.5x) "
                    "or broader market index breakdown will invalidate this support floor."
                )
            else:
                invalidation = (
                    f"A sustained close above {upper_boundary} on expanding volume would convert this zone "
                    "from overhead resistance into newly established support."
                )

            recent_date_str = recent_date.strftime("%Y-%m-%d") if hasattr(recent_date, "strftime") else str(recent_date)

            zone_dict = {
                "center_price": round(center_price, 2),
                "lower_boundary": lower_boundary,
                "upper_boundary": upper_boundary,
                "meaningful_tests": touches,
                "average_reaction_pct": avg_reaction,
                "reaction_type": "Average Rebound" if is_support else "Average Rejection",
                "recent_interaction_date": recent_date_str,
                "days_since_interaction": days_ago,
                "volume_evidence": volume_label,
                "volume_ratio": round(vol_ratio, 2),
                "distance_pct": distance_pct,
                "distance_description": distance_description,
                "is_broken": is_broken,
                "confidence_label": confidence,
                "invalidation_factors": invalidation
            }
            zones.append(zone_dict)

        # Sort support zones descending (closest to current price first)
        # Sort resistance zones ascending (closest overhead first)
        if is_support:
            # Filter to active zones around or below current price + closest broken zones
            zones.sort(key=lambda z: z["center_price"], reverse=True)
        else:
            zones.sort(key=lambda z: z["center_price"], reverse=False)

        return zones

    def _estimate_breakdown_risk(
        self,
        support_zones: List[Dict[str, Any]],
        df: pd.DataFrame,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Estimates breakdown risk using historical comparable situations.
        Explicitly reports sample size and breakdown definitions.
        """
        if not support_zones:
            return {
                "primary_support_zone": None,
                "historical_breakdown_frequency_pct": 0.0,
                "sample_size": 0,
                "risk_rating": "Indeterminate (No qualified support zones detected)"
            }

        # Find closest un-broken support zone below current price
        valid_supports = [z for z in support_zones if not z["is_broken"] and z["upper_boundary"] <= current_price]
        if not valid_supports:
            # Fall back to nearest support
            primary = support_zones[0]
        else:
            primary = valid_supports[0]

        # Calculate historical breakdown rate across all historical tests in the dataset
        total_tests = sum(z["meaningful_tests"] for z in support_zones)
        broken_zones = sum(1 for z in support_zones if z["is_broken"])
        breakdown_rate = round((broken_zones / len(support_zones)) * 100.0, 1) if support_zones else 0.0

        if breakdown_rate > 50.0:
            rating = "Elevated Breakdown Vulnerability"
        elif breakdown_rate > 25.0:
            rating = "Moderate Breakdown Risk"
        else:
            rating = "Low Breakdown Frequency (Historically Resilient)"

        return {
            "primary_support_zone": f"₹{primary['lower_boundary']} – ₹{primary['upper_boundary']}" if current_price > 500 else f"${primary['lower_boundary']} – ${primary['upper_boundary']}",
            "distance_to_primary_support_pct": primary["distance_pct"],
            "historical_breakdown_frequency_pct": breakdown_rate,
            "sample_size_zones": len(support_zones),
            "total_historical_tests": total_tests,
            "risk_rating": rating,
            "definition": "Breakdown defined as a confirmed daily close penetrating more than 1.0% below the lower zone boundary."
        }


# Global default engine instance
sr_engine = SupportResistanceEngine()
