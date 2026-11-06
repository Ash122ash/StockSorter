"""
StockSentry AI - News Intelligence & Financial Sentiment Engine
Retrieves genuine company news via Yahoo Finance / NewsAPI, eliminates syndication duplicates,
applies finance-specific event taxonomy, calculates contextual sentiment, and weights aggregate impact.
"""

import re
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import yfinance as yf
import requests
from stocksentry.config import NEWS_API_KEY
from stocksentry.storage import storage

logger = logging.getLogger(__name__)

# Financial event classification keyword taxonomies
EVENT_TAXONOMY = {
    "Earnings & Financial Results": {
        "keywords": ["quarterly", "q1", "q2", "q3", "q4", "profit", "net profit", "revenue", "ebitda", "margin", "earnings", "pat", "loss"],
        "base_importance": "High"
    },
    "Regulatory & Legal": {
        "keywords": ["sebi", "rbi", "sec", "investigation", "probe", "lawsuit", "litigation", "court", "penalty", "fine", "raid", "subpoena"],
        "base_importance": "High"
    },
    "Mergers & Acquisitions": {
        "keywords": ["acquisition", "merger", "takeover", "buys stake", "divestment", "stake sale", "amalgamation"],
        "base_importance": "High"
    },
    "Debt & Liquidity": {
        "keywords": ["debt", "refinancing", "default", "credit rating", "downgrade", "liquidity", "bond issue", "ncd"],
        "base_importance": "High"
    },
    "Major Contracts & Orders": {
        "keywords": ["order win", "contract", "deal", "agreement", "mou", "partnership", "awarded"],
        "base_importance": "Medium"
    },
    "Executive & Governance": {
        "keywords": ["ceo", "cfo", "md", "director", "resigns", "appointed", "promoter", "board meeting", "auditor"],
        "base_importance": "Medium"
    },
    "Guidance & Analyst Ratings": {
        "keywords": ["target price", "upgrade", "downgrade", "guidance", "outlook", "bullish", "bearish", "brokerage"],
        "base_importance": "Medium"
    }
}

# Financial domain sentiment lexicon with calibrated directional polarity
FIN_POSITIVE_LEXICON = {
    "surge": 1.8, "rally": 1.6, "jump": 1.5, "record": 1.4, "beat": 1.7, "outperform": 1.6,
    "growth": 1.2, "profit": 1.5, "upgrade": 1.6, "gain": 1.1, "bullish": 1.4, "dividend": 1.1,
    "expansion": 1.2, "order win": 1.8, "approved": 1.3, "soars": 1.7, "robust": 1.4, "exceeds": 1.5
}

FIN_NEGATIVE_LEXICON = {
    "plunge": -2.0, "crash": -2.2, "slump": -1.8, "tumble": -1.8, "probe": -1.9, "fall": -1.2,
    "drop": -1.2, "loss": -1.6, "downgrade": -1.8, "penalty": -1.9, "fine": -1.7, "litigation": -1.6,
    "debt": -1.1, "default": -2.5, "fraud": -2.8, "resigns": -1.4, "subpoena": -1.9, "miss": -1.5,
    "weak": -1.3, "bearish": -1.4, "decline": -1.2, "scrutiny": -1.5, "raid": -2.2
}


class NewsIntelligenceEngine:
    """Production News Intelligence Engine with fallback transparency."""

    def __init__(self, api_key: str = NEWS_API_KEY):
        self.api_key = api_key

    def fetch_and_analyze(self, symbol: str, company_name: str = "") -> Dict[str, Any]:
        """
        Retrieves company-specific news, processes sentiment, deduplicates syndication,
        and compiles an aggregate market sentiment rating.
        """
        clean_sym = symbol.strip().upper()

        # Check cache
        cached_articles = storage.get_news(clean_sym)
        if cached_articles is not None:
            return self._compile_news_summary(clean_sym, cached_articles, from_cache=True)

        raw_articles = []

        # 1. Primary Source: Yahoo Finance ticker news (Real, live, ticker-targeted)
        try:
            ticker = yf.Ticker(clean_sym)
            yf_news = getattr(ticker, "news", None)
            if yf_news and isinstance(yf_news, list):
                for item in yf_news:
                    title = item.get("title", "")
                    if not title:
                        continue
                    publisher = item.get("publisher", "Financial News Wire")
                    link = item.get("link", "#")
                    pub_time = item.get("providerPublishTime", int(datetime.now().timestamp()))
                    dt_str = datetime.fromtimestamp(pub_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

                    raw_articles.append({
                        "headline": title,
                        "publisher": publisher,
                        "link": link,
                        "pub_timestamp": pub_time,
                        "publication_time": dt_str,
                        "raw_summary": item.get("summary", "")
                    })
        except Exception as e:
            logger.warning(f"Yahoo Finance news fetch failed for {clean_sym}: {e}")

        # 2. Secondary Source: NewsAPI if configured
        if self.api_key and len(raw_articles) < 5:
            try:
                query = clean_sym.split(".")[0]
                url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=10&apiKey={self.api_key}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for art in data.get("articles", []):
                        raw_articles.append({
                            "headline": art.get("title", ""),
                            "publisher": art.get("source", {}).get("name", "NewsAPI"),
                            "link": art.get("url", "#"),
                            "pub_timestamp": int(datetime.fromisoformat(art.get("publishedAt").replace("Z", "+00:00")).timestamp()),
                            "publication_time": art.get("publishedAt", ""),
                            "raw_summary": art.get("description", "")
                        })
            except Exception as e:
                logger.warning(f"NewsAPI query failed: {e}")

        # Deduplicate and analyze articles
        processed_articles = self._process_articles(clean_sym, raw_articles)

        # Cache results
        storage.save_news(clean_sym, processed_articles)

        return self._compile_news_summary(clean_sym, processed_articles, from_cache=False)

    def _process_articles(self, symbol: str, raw_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_titles = set()
        cleaned_articles = []
        base_ticker = symbol.split(".")[0].lower()

        for art in raw_articles:
            headline = art["headline"]
            # Deduplication key: alphanumeric normalized headline snippet
            norm_key = re.sub(r"[^a-zA-Z0-9]", "", headline.lower()[:40])
            if norm_key in seen_titles:
                continue
            seen_titles.add(norm_key)

            text_to_score = f"{headline} {art.get('raw_summary', '')}".lower()

            # Event Categorization
            event_category = "General Corporate / Market Developments"
            importance = "Low"
            for cat, details in EVENT_TAXONOMY.items():
                if any(kw in text_to_score for kw in details["keywords"]):
                    event_category = cat
                    importance = details["base_importance"]
                    break

            # Company Relevance Scoring
            relevance = "Direct Company Specific" if base_ticker in text_to_score else "Sector / Macro Relevant"

            # Financial Lexicon Sentiment Scoring
            score = 0.0
            for pos_word, weight in FIN_POSITIVE_LEXICON.items():
                if pos_word in text_to_score:
                    score += weight
            for neg_word, weight in FIN_NEGATIVE_LEXICON.items():
                if neg_word in text_to_score:
                    score += weight

            # Normalize to [-1.0, 1.0]
            normalized_score = max(-1.0, min(1.0, score / 3.0))

            if normalized_score > 0.2:
                sentiment_label = "Positive"
            elif normalized_score < -0.2:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

            # Generate context rationale
            explanation = self._generate_explanation(event_category, sentiment_label, headline)

            cleaned_articles.append({
                "headline": headline,
                "publisher": art["publisher"],
                "publication_time": art["publication_time"],
                "pub_timestamp": art["pub_timestamp"],
                "link": art["link"],
                "relevance": relevance,
                "sentiment_label": sentiment_label,
                "sentiment_score": round(normalized_score, 2),
                "event_category": event_category,
                "importance": importance,
                "explanation": explanation
            })

        return cleaned_articles

    def _generate_explanation(self, category: str, sentiment: str, headline: str) -> str:
        if "Earnings" in category:
            if sentiment == "Positive":
                return "Strong financial metrics support earnings expansion; monitor if already discounted by recent run-up."
            elif sentiment == "Negative":
                return "Profit contraction or margin erosion creates direct downward valuation pressure."
            return "Financial numbers inline; market attention turns to management commentary and forward guidance."
        elif "Regulatory" in category:
            return "Regulatory scrutiny introduces legal uncertainty, potential fines, and immediate risk-off sentiment."
        elif "Mergers" in category:
            return "M&A brings operational scale but creates execution, integration, and short-term debt risk."
        elif "Debt" in category:
            return "Credit/leverage developments directly influence balance-sheet risk and institutional borrowing costs."
        elif "Contracts" in category:
            return "New revenue stream enhances forward order backlog visibility."
        else:
            return f"Market development carrying {sentiment.lower()} headline tone; review price reaction at key technical levels."

    def _compile_news_summary(self, symbol: str, articles: List[Dict[str, Any]], from_cache: bool) -> Dict[str, Any]:
        api_notice = None
        if not self.api_key:
            api_notice = "Using live Yahoo Finance news feed. Configure NEWS_API_KEY in .env for expanded multi-publisher aggregation."

        if not articles:
            return {
                "symbol": symbol,
                "articles_count": 0,
                "articles": [],
                "aggregate_sentiment": "Neutral / No Recent Headlines",
                "aggregate_score": 0.0,
                "high_impact_events": [],
                "from_cache": from_cache,
                "api_notice": api_notice
            }

        # Calculate time-weighted and importance-weighted aggregate sentiment
        now_ts = int(datetime.now().timestamp())
        weighted_sum = 0.0
        total_weight = 0.0
        high_impact = []

        for art in articles:
            age_hours = max(1.0, (now_ts - art["pub_timestamp"]) / 3600.0)
            time_decay = 1.0 / (1.0 + (age_hours / 48.0)) # 48h decay
            imp_multiplier = 2.0 if art["importance"] == "High" else (1.3 if art["importance"] == "Medium" else 1.0)
            
            weight = time_decay * imp_multiplier
            weighted_sum += art["sentiment_score"] * weight
            total_weight += weight

            if art["importance"] == "High" or abs(art["sentiment_score"]) >= 0.5:
                high_impact.append(art)

        agg_score = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0

        if agg_score >= 0.25:
            overall_sentiment = "Bullish / Favorable News Flow"
        elif agg_score <= -0.25:
            overall_sentiment = "Bearish / Unfavorable Headwinds"
        else:
            overall_sentiment = "Neutral / Balanced Sentiment"

        # Sort articles by pub_timestamp descending
        articles_sorted = sorted(articles, key=lambda x: x["pub_timestamp"], reverse=True)

        return {
            "symbol": symbol,
            "articles_count": len(articles_sorted),
            "articles": articles_sorted,
            "aggregate_sentiment": overall_sentiment,
            "aggregate_score": agg_score,
            "high_impact_events": high_impact[:3],
            "from_cache": from_cache,
            "api_notice": api_notice
        }


# Global default news engine
news_engine = NewsIntelligenceEngine()
