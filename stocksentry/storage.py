"""
StockSentry AI - Storage Layer (SQLite initially, PostgreSQL-ready architecture)
"""

import sqlite3
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from stocksentry.config import DB_PATH

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage interface for market cache, news, and validation runs."""

    @abstractmethod
    def save_ohlc_cache(self, symbol: str, period: str, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_ohlc_cache(self, symbol: str, period: str, max_age_seconds: int = 900) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_news(self, symbol: str, articles: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def get_news(self, symbol: str, max_age_seconds: int = 1800) -> Optional[List[Dict[str, Any]]]:
        pass

    @abstractmethod
    def log_validation_run(self, symbol: str, model_name: str, metrics: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_validation_history(self, symbol: str) -> List[Dict[str, Any]]:
        pass


class SQLiteStorage(StorageBackend):
    """Thread-safe SQLite storage implementation."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # OHLC Cache Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlc_cache (
                    symbol TEXT NOT NULL,
                    period TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, period)
                )
            """)
            # News Cache Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_cache (
                    symbol TEXT NOT NULL,
                    articles_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol)
                )
            """)
            # Model Validation Logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Query Audit Logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_ohlc_cache(self, symbol: str, period: str, data: Dict[str, Any]) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ohlc_cache (symbol, period, data_json, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(symbol, period) DO UPDATE SET
                        data_json = excluded.data_json,
                        updated_at = CURRENT_TIMESTAMP
                """, (symbol.upper(), period, json.dumps(data)))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving OHLC cache for {symbol}: {e}")

    def get_ohlc_cache(self, symbol: str, period: str, max_age_seconds: int = 900) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data_json, strftime('%s', 'now') - strftime('%s', updated_at) AS age
                    FROM ohlc_cache
                    WHERE symbol = ? AND period = ?
                """, (symbol.upper(), period))
                row = cursor.fetchone()
                if row and row["age"] <= max_age_seconds:
                    return json.loads(row["data_json"])
        except Exception as e:
            logger.error(f"Error reading OHLC cache for {symbol}: {e}")
        return None

    def save_news(self, symbol: str, articles: List[Dict[str, Any]]) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO news_cache (symbol, articles_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(symbol) DO UPDATE SET
                        articles_json = excluded.articles_json,
                        updated_at = CURRENT_TIMESTAMP
                """, (symbol.upper(), json.dumps(articles)))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving news cache for {symbol}: {e}")

    def get_news(self, symbol: str, max_age_seconds: int = 1800) -> Optional[List[Dict[str, Any]]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT articles_json, strftime('%s', 'now') - strftime('%s', updated_at) AS age
                    FROM news_cache
                    WHERE symbol = ?
                """, (symbol.upper(),))
                row = cursor.fetchone()
                if row and row["age"] <= max_age_seconds:
                    return json.loads(row["articles_json"])
        except Exception as e:
            logger.error(f"Error reading news cache for {symbol}: {e}")
        return None

    def log_validation_run(self, symbol: str, model_name: str, metrics: Dict[str, Any]) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO validation_runs (symbol, model_name, metrics_json)
                    VALUES (?, ?, ?)
                """, (symbol.upper(), model_name, json.dumps(metrics)))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging validation run for {symbol}: {e}")

    def get_validation_history(self, symbol: str) -> List[Dict[str, Any]]:
        runs = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT model_name, metrics_json, created_at
                    FROM validation_runs
                    WHERE symbol = ?
                    ORDER BY id DESC LIMIT 10
                """, (symbol.upper(),))
                for row in cursor.fetchall():
                    runs.append({
                        "model_name": row["model_name"],
                        "metrics": json.loads(row["metrics_json"]),
                        "created_at": row["created_at"]
                    })
        except Exception as e:
            logger.error(f"Error reading validation history for {symbol}: {e}")
        return runs

    def log_audit(self, symbol: str, action: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO audit_logs (symbol, action) VALUES (?, ?)", (symbol.upper(), action))
                conn.commit()
        except Exception:
            pass


# Global singleton storage instance
storage = SQLiteStorage()
