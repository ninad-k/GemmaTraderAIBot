from pathlib import Path

import gemma_trader.safety as safety


def test_drawdown_breaker_trips(monkeypatch, tmp_path):
    monkeypatch.setattr(safety, "STATE_PATH", tmp_path / "s.json")
    monkeypatch.setattr(safety, "_singleton", None)
    s = safety.SafetyController(max_drawdown_pct=10.0)
    s.update_equity(100_000)
    assert s.is_halted() is False
    s.update_equity(85_000)  # 15% drawdown
    assert s.is_halted() is True
    assert s.state.breaker_tripped is True


def test_halt_resume_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(safety, "STATE_PATH", tmp_path / "s.json")
    s = safety.SafetyController()
    s.halt("test")
    assert s.is_halted()
    s.resume()
    assert not s.is_halted()


def test_heartbeat_seconds(monkeypatch, tmp_path):
    monkeypatch.setattr(safety, "STATE_PATH", tmp_path / "s.json")
    s = safety.SafetyController()
    s.heartbeat()
    assert s.seconds_since_heartbeat() < 5
