from datetime import datetime, timedelta
from pathlib import Path

from ensemble import ensemble_decide, prompt_hash, FeatureDedupeCache
from news_calendar import NewsCalendar


def test_prompt_hash_is_stable():
    a = prompt_hash("system", "user")
    b = prompt_hash("system", "user")
    c = prompt_hash("system", "user2")
    assert a == b and a != c and len(a) == 12


def test_dedupe_cache_hits():
    c = FeatureDedupeCache(ttl_seconds=60)
    md = {"symbol": "BTCUSD", "timeframe": "1m", "rsi": 30.1, "macd_hist": 0.01,
          "trend": "UP", "ichimoku_signal": "BULL", "ema_cross": "GOLDEN", "adx": 22}
    assert c.lookup(md) is None
    c.store(md, {"action": "BUY", "confidence": 0.7})
    hit = c.lookup({**md, "rsi": 30.4})  # still in same quantised bucket (step=2)
    assert hit is not None
    assert hit.get("cache_hit") is True


def test_ensemble_disabled_passthrough():
    cfg = {"ollama": {"model": "m"}, "ensemble": {"enabled": False}}
    calls = []
    def fn(md, c):
        calls.append(c["ollama"]["model"])
        return {"action": "BUY", "confidence": 0.9}
    d = ensemble_decide({"symbol": "X"}, cfg, fn)
    assert d["action"] == "BUY"
    assert calls == ["m"]


def test_ensemble_agreement_gate():
    cfg = {"ollama": {"model": "m1"},
           "ensemble": {"enabled": True, "models": ["a", "b"], "min_agreement": 2}}

    def disagree(md, c):
        return {"action": "BUY" if c["ollama"]["model"] == "a" else "SELL", "confidence": 0.8}

    d = ensemble_decide({"symbol": "X"}, cfg, disagree)
    assert d["action"] == "HOLD"

    def agree(md, c):
        return {"action": "BUY", "confidence": 0.8}
    d2 = ensemble_decide({"symbol": "X"}, cfg, agree)
    assert d2["action"] == "BUY"
    assert d2["ensemble_actions"] == ["BUY", "BUY"]


def test_news_blackout(tmp_path: Path):
    p = tmp_path / "n.yaml"
    nc = NewsCalendar(path=p)
    now = datetime.utcnow()
    nc.add((now - timedelta(minutes=5)).isoformat(),
           (now + timedelta(minutes=5)).isoformat(), "fomc")
    in_bl, label = nc.in_blackout(now)
    assert in_bl is True and label == "fomc"
    # outside
    in_bl2, _ = nc.in_blackout(now + timedelta(hours=1))
    assert in_bl2 is False
