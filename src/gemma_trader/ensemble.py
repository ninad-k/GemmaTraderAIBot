"""
Multi-model ensemble + prompt-version hashing + feature-dedupe cache.

- ensemble_decide: calls N Ollama models in sequence (config:ensemble.models).
  Returns the agreed action + averaged confidence; HOLD if models disagree.
- prompt_hash: sha1 of SYSTEM_PROMPT + adaptive context, logged per decision
  so we can tie every trade to the exact prompt version that produced it.
- FeatureDedupeCache: short-TTL cache keyed on a quantised feature bucket;
  avoids redundant LLM calls when candles barely move.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from statistics import mean
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def prompt_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def ensemble_decide(
    market_data: dict,
    config: dict,
    analyze_fn: Callable[[dict, dict], dict],
) -> dict:
    """
    Run `analyze_fn` against each model in config['ensemble']['models'] and
    return the agreed decision.
    """
    ens = (config.get("ensemble") or {})
    if not ens.get("enabled"):
        return analyze_fn(market_data, config)

    models = ens.get("models") or [config["ollama"]["model"]]
    min_agree = int(ens.get("min_agreement", len(models)))
    decisions: list[dict] = []

    for m in models:
        cfg_override = dict(config)
        cfg_override["ollama"] = {**config["ollama"], "model": m}
        try:
            d = analyze_fn(market_data, cfg_override)
            decisions.append(d)
        except Exception as e:
            logger.warning(f"ensemble: model {m} failed: {e}")

    if not decisions:
        return {"action": "HOLD", "confidence": 0.0, "reason": "ensemble: no models responded"}

    actions = [str(d.get("action", "HOLD")).upper() for d in decisions]
    most = max(set(actions), key=actions.count)
    agree_count = actions.count(most)
    if agree_count < min_agree or most == "HOLD":
        return {
            "action": "HOLD", "confidence": 0.0,
            "reason": f"ensemble disagreement ({agree_count}/{len(decisions)} for {most})",
            "ensemble_actions": actions,
        }

    confs = [float(d.get("confidence", 0) or 0) for d in decisions if str(d.get("action")).upper() == most]
    out = dict(decisions[0])
    out["action"] = most
    out["confidence"] = round(mean(confs), 3) if confs else 0.0
    out["ensemble_actions"] = actions
    out["ensemble_models"] = models
    return out


class FeatureDedupeCache:
    """
    Cache keyed on a coarse bucket of the most important features.
    If the same bucket re-appears within ttl seconds we reuse the previous
    decision and skip the LLM call entirely.
    """

    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, dict]] = {}

    @staticmethod
    def _bucket(md: dict) -> str:
        def q(x, step):
            try:
                return round(float(x) / step) * step
            except Exception:
                return None
        parts = [
            md.get("symbol"),
            md.get("timeframe"),
            q(md.get("rsi"), 2),
            q(md.get("macd_hist"), 0.05),
            md.get("trend"),
            md.get("ichimoku_signal"),
            md.get("ema_cross"),
            q(md.get("adx"), 2),
        ]
        return "|".join("" if p is None else str(p) for p in parts)

    def lookup(self, md: dict) -> Optional[dict]:
        key = self._bucket(md)
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            if entry and now - entry[0] < self.ttl:
                cached = dict(entry[1])
                cached["cache_hit"] = True
                return cached
        return None

    def store(self, md: dict, decision: dict) -> None:
        key = self._bucket(md)
        with self._lock:
            self._cache[key] = (time.time(), dict(decision))


_singleton: Optional[FeatureDedupeCache] = None


def get_dedupe_cache(ttl_seconds: int = 60) -> FeatureDedupeCache:
    global _singleton
    if _singleton is None:
        _singleton = FeatureDedupeCache(ttl_seconds)
    return _singleton


