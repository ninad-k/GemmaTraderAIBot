"""
SQLite Storage Layer
=====================
Durable storage for trade outcomes, decisions, and OHLCV cache.
Replaces the JSON-file scheme which doesn't scale past ~500 records.

Thread-safe via connection-per-operation pattern (SQLite's built-in locking).
Read layer falls back to JSON if DB not yet migrated — zero-downtime upgrade.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    action TEXT,
    entry_price REAL,
    close_price REAL,
    sl REAL,
    tp REAL,
    qty REAL,
    profit REAL,
    result TEXT,
    confidence REAL,
    reason TEXT,
    entry_time TEXT,
    close_time TEXT,
    duration_minutes REAL,
    indicators_snapshot TEXT,
    regime TEXT,
    prompt_hash TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_outcomes_symbol ON trade_outcomes(symbol);
CREATE INDEX IF NOT EXISTS idx_outcomes_close_time ON trade_outcomes(close_time);
CREATE INDEX IF NOT EXISTS idx_outcomes_regime ON trade_outcomes(regime);

CREATE TABLE IF NOT EXISTS gemma_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT,
    confidence REAL,
    reason TEXT,
    prompt_hash TEXT,
    raw_response TEXT,
    indicators_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON gemma_decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON gemma_decisions(timestamp);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE,
    symbol TEXT NOT NULL,
    action TEXT,
    qty REAL,
    sl REAL,
    tp REAL,
    status TEXT,
    timestamp TEXT,
    closed INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades(closed);

CREATE TABLE IF NOT EXISTS ohlcv_cache (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup ON ohlcv_cache(symbol, timeframe, timestamp);

CREATE TABLE IF NOT EXISTS parameter_adjustments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    parameter TEXT,
    old_value REAL,
    new_value REAL,
    reason TEXT
);
"""


class TradingDB:
    """Thread-safe SQLite wrapper for trading data."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            isolation_level=None,  # autocommit
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    # ── Trade outcomes ──

    def record_outcome(self, outcome: dict) -> int:
        """Insert a trade outcome; returns row id."""
        with self._lock, self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trade_outcomes
                (symbol, action, entry_price, close_price, sl, tp, qty, profit,
                 result, confidence, reason, entry_time, close_time,
                 duration_minutes, indicators_snapshot, regime, prompt_hash)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    outcome.get("symbol"),
                    outcome.get("action"),
                    outcome.get("entry_price"),
                    outcome.get("close_price"),
                    outcome.get("sl"),
                    outcome.get("tp"),
                    outcome.get("qty"),
                    outcome.get("profit"),
                    outcome.get("result"),
                    outcome.get("confidence"),
                    outcome.get("reason"),
                    outcome.get("entry_time"),
                    outcome.get("close_time"),
                    outcome.get("duration_minutes"),
                    json.dumps(outcome.get("indicators_snapshot", {})),
                    outcome.get("regime"),
                    outcome.get("prompt_hash"),
                ),
            )
            return cursor.lastrowid

    def query_outcomes(
        self,
        symbol: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        regime: Optional[str] = None,
        limit: int = 10000,
    ) -> list[dict]:
        """Query outcomes with optional filters."""
        query = "SELECT * FROM trade_outcomes WHERE 1=1"
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start:
            query += " AND close_time >= ?"
            params.append(start)
        if end:
            query += " AND close_time <= ?"
            params.append(end)
        if regime:
            query += " AND regime = ?"
            params.append(regime)
        query += " ORDER BY close_time DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            # Deserialize JSON fields
            if d.get("indicators_snapshot"):
                try:
                    d["indicators_snapshot"] = json.loads(d["indicators_snapshot"])
                except Exception:
                    d["indicators_snapshot"] = {}
            result.append(d)
        return result

    def count_outcomes(self, symbol: Optional[str] = None) -> int:
        with self._conn() as conn:
            if symbol:
                return conn.execute(
                    "SELECT COUNT(*) FROM trade_outcomes WHERE symbol = ?",
                    (symbol,),
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]

    # ── Decisions ──

    def record_decision(self, decision: dict) -> int:
        with self._lock, self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO gemma_decisions
                (timestamp, symbol, action, confidence, reason, prompt_hash,
                 raw_response, indicators_summary)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    decision.get("timestamp", datetime.now().isoformat()),
                    decision.get("symbol", "UNKNOWN"),
                    decision.get("action"),
                    decision.get("confidence"),
                    decision.get("reason"),
                    decision.get("prompt_hash"),
                    decision.get("raw_gemma_response", ""),
                    json.dumps(decision.get("indicators_summary", {})),
                ),
            )
            return cursor.lastrowid

    def query_decisions(
        self,
        symbol: Optional[str] = None,
        limit: int = 2000,
    ) -> list[dict]:
        query = "SELECT * FROM gemma_decisions WHERE 1=1"
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    # ── OHLCV cache ──

    def cache_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict],
    ) -> int:
        """Insert OHLCV candles; ignores duplicates via PRIMARY KEY."""
        if not candles:
            return 0
        with self._lock, self._conn() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO ohlcv_cache
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        symbol,
                        timeframe,
                        c["timestamp"],
                        c["open"],
                        c["high"],
                        c["low"],
                        c["close"],
                        c.get("volume", 0),
                    )
                    for c in candles
                ],
            )
            return cursor.rowcount

    def query_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_cache
                WHERE symbol = ? AND timeframe = ?
                  AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (symbol, timeframe, start, end),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Migration from JSON ──

    def migrate_from_json(self, logs_dir: Path) -> dict:
        """Migrate existing JSON log files into SQLite. Returns count per table."""
        counts = {"outcomes": 0, "decisions": 0}

        outcomes_path = logs_dir / "trade_outcomes.json"
        if outcomes_path.exists():
            try:
                outcomes = json.loads(outcomes_path.read_text(encoding="utf-8-sig"))
                for o in outcomes:
                    self.record_outcome(o)
                counts["outcomes"] = len(outcomes)
                logger.info(f"Migrated {len(outcomes)} outcomes")
            except Exception as e:
                logger.warning(f"Failed to migrate outcomes: {e}")

        decisions_path = logs_dir / "gemma_decisions.json"
        if decisions_path.exists():
            try:
                decisions = json.loads(decisions_path.read_text(encoding="utf-8-sig"))
                for d in decisions:
                    self.record_decision(d)
                counts["decisions"] = len(decisions)
                logger.info(f"Migrated {len(decisions)} decisions")
            except Exception as e:
                logger.warning(f"Failed to migrate decisions: {e}")

        return counts


# ── Singleton accessor ──

_db_instance: Optional[TradingDB] = None


def get_db(db_path: Optional[Path] = None) -> TradingDB:
    """Get or create the singleton TradingDB instance."""
    global _db_instance
    if _db_instance is None:
        if db_path is None:
            db_path = Path("logs/trading.db")
        _db_instance = TradingDB(db_path)
    return _db_instance


def reset_db() -> None:
    """Reset the singleton (mainly for tests)."""
    global _db_instance
    _db_instance = None


# ── CLI ──

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading DB management")
    parser.add_argument("--migrate", action="store_true", help="Migrate JSON → SQLite")
    parser.add_argument("--db", default="logs/trading.db", help="DB path")
    parser.add_argument("--logs", default="logs", help="Logs directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    db = TradingDB(Path(args.db))

    if args.migrate:
        counts = db.migrate_from_json(Path(args.logs))
        print(f"✓ Migrated: {counts}")
    else:
        print(f"DB initialized at {args.db}")
        print(f"Outcomes: {db.count_outcomes()}")
