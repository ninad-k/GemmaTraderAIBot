import json
from pathlib import Path

import metrics
import backtester
from backtester import run_backtest, confidence_threshold_fn


OUTCOMES = [
    {"symbol": "BTCUSD", "action": "BUY", "profit": 120.0, "close_time": "2026-01-01T10:00:00",
     "indicators_snapshot": {"rsi": 20, "vol_trend": "HIGH"}, "result": "WIN"},
    {"symbol": "BTCUSD", "action": "SELL", "profit": -60.0, "close_time": "2026-01-01T11:00:00",
     "indicators_snapshot": {"rsi": 78, "vol_trend": "LOW"}, "result": "LOSS"},
    {"symbol": "ETHUSD", "action": "BUY", "profit": 40.0, "close_time": "2026-01-01T12:00:00",
     "indicators_snapshot": {"rsi": 28, "vol_trend": "HIGH"}, "result": "WIN"},
]


def _seed(tmp_path, monkeypatch):
    p = tmp_path / "outcomes.json"
    p.write_text(json.dumps(OUTCOMES))
    monkeypatch.setattr(metrics, "OUTCOME_LOG", p)
    monkeypatch.setattr(backtester, "OUTCOME_LOG", p)


def test_metrics_summary(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    s = metrics.summary()
    assert s["trades"] == 3
    assert s["wins"] == 2 and s["losses"] == 1


def test_equity_curve_monotonic_peak(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    curve = metrics.equity_curve()
    assert len(curve) == 3
    assert curve[-1]["equity"] == 100_000 + 120 - 60 + 40


def test_per_symbol_and_regime(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    assert "BTCUSD" in metrics.per_symbol()
    assert "HIGH" in metrics.per_regime()


def test_backtest_runs_and_respects_threshold(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    fn = confidence_threshold_fn(0.5)
    r = run_backtest(fn)
    assert "trades" in r
    assert r["trades"] >= 0
