"""Tests for SQLite storage layer."""

import json
import pytest
from pathlib import Path

from gemma_trader.storage import TradingDB, reset_db


@pytest.fixture
def db(tmp_path):
    reset_db()
    return TradingDB(tmp_path / "test.db")


def test_schema_created(db):
    with db._conn() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
    assert "trade_outcomes" in tables
    assert "gemma_decisions" in tables
    assert "ohlcv_cache" in tables


def test_record_outcome(db):
    outcome = {
        "symbol": "BTCUSD",
        "action": "BUY",
        "entry_price": 45000.0,
        "close_price": 45500.0,
        "qty": 0.1,
        "profit": 50.0,
        "result": "WIN",
        "confidence": 0.75,
        "entry_time": "2024-01-01T10:00:00",
        "close_time": "2024-01-01T10:30:00",
        "indicators_snapshot": {"rsi": 35.0, "macd": 0.02},
    }
    row_id = db.record_outcome(outcome)
    assert row_id > 0

    assert db.count_outcomes() == 1
    assert db.count_outcomes(symbol="BTCUSD") == 1
    assert db.count_outcomes(symbol="NONEXISTENT") == 0


def test_query_outcomes_with_filters(db):
    for i, symbol in enumerate(["BTCUSD", "ETHUSD", "BTCUSD"]):
        db.record_outcome({
            "symbol": symbol,
            "action": "BUY",
            "profit": 10.0 * i,
            "result": "WIN" if i % 2 == 0 else "LOSS",
            "close_time": f"2024-01-0{i+1}T10:00:00",
        })

    btc_outcomes = db.query_outcomes(symbol="BTCUSD")
    assert len(btc_outcomes) == 2

    all_outcomes = db.query_outcomes()
    assert len(all_outcomes) == 3


def test_indicators_snapshot_serialization(db):
    db.record_outcome({
        "symbol": "BTCUSD",
        "indicators_snapshot": {"rsi": 50.0, "nested": {"a": 1}},
    })
    results = db.query_outcomes()
    assert results[0]["indicators_snapshot"]["rsi"] == 50.0
    assert results[0]["indicators_snapshot"]["nested"]["a"] == 1


def test_record_decision(db):
    decision = {
        "timestamp": "2024-01-01T10:00:00",
        "symbol": "BTCUSD",
        "action": "BUY",
        "confidence": 0.75,
        "reason": "test",
        "prompt_hash": "abc123",
    }
    row_id = db.record_decision(decision)
    assert row_id > 0

    decisions = db.query_decisions()
    assert len(decisions) == 1
    assert decisions[0]["symbol"] == "BTCUSD"


def test_ohlcv_cache(db):
    candles = [
        {
            "timestamp": "2024-01-01T00:00:00",
            "open": 45000, "high": 45100, "low": 44900,
            "close": 45050, "volume": 1234,
        },
        {
            "timestamp": "2024-01-01T01:00:00",
            "open": 45050, "high": 45200, "low": 45000,
            "close": 45150, "volume": 1500,
        },
    ]
    inserted = db.cache_ohlcv("BTCUSD", "1h", candles)
    assert inserted == 2

    # Duplicate insert should not double-count
    again = db.cache_ohlcv("BTCUSD", "1h", candles)
    assert again == 0

    retrieved = db.query_ohlcv("BTCUSD", "1h", "2024-01-01", "2024-01-02")
    assert len(retrieved) == 2
    assert retrieved[0]["open"] == 45000


def test_migrate_from_json(db, tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    outcomes = [
        {"symbol": "BTCUSD", "action": "BUY", "result": "WIN", "profit": 10.0},
        {"symbol": "ETHUSD", "action": "SELL", "result": "LOSS", "profit": -5.0},
    ]
    (logs_dir / "trade_outcomes.json").write_text(json.dumps(outcomes))

    decisions = [
        {"timestamp": "2024-01-01T10:00:00", "symbol": "BTCUSD", "action": "BUY"},
    ]
    (logs_dir / "gemma_decisions.json").write_text(json.dumps(decisions))

    counts = db.migrate_from_json(logs_dir)
    assert counts["outcomes"] == 2
    assert counts["decisions"] == 1
    assert db.count_outcomes() == 2


def test_migrate_missing_files(db, tmp_path):
    """Migration is gracious with missing files."""
    logs_dir = tmp_path / "empty_logs"
    logs_dir.mkdir()

    counts = db.migrate_from_json(logs_dir)
    assert counts["outcomes"] == 0
    assert counts["decisions"] == 0


def test_thread_safety_sequential_writes(db):
    """Many sequential writes should all persist."""
    for i in range(50):
        db.record_outcome({
            "symbol": "BTCUSD",
            "action": "BUY",
            "profit": float(i),
        })
    assert db.count_outcomes() == 50


def test_query_with_regime(db):
    db.record_outcome({
        "symbol": "BTCUSD",
        "result": "WIN",
        "regime": "trending_up",
        "close_time": "2024-01-01",
    })
    db.record_outcome({
        "symbol": "BTCUSD",
        "result": "LOSS",
        "regime": "high_vol_chop",
        "close_time": "2024-01-02",
    })

    trending = db.query_outcomes(regime="trending_up")
    assert len(trending) == 1
    assert trending[0]["regime"] == "trending_up"
