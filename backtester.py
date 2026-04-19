"""
Backtester
===========
Replays the indicator snapshots stored in logs/trade_outcomes.json
against an arbitrary decision function. Useful for offline A/B of a
new prompt or model without risking real capital.

Usage:
    from backtester import run_backtest
    result = run_backtest(decision_fn=my_fn)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

OUTCOME_LOG = Path("logs/trade_outcomes.json")


def _load_outcomes() -> list[dict]:
    if not OUTCOME_LOG.exists():
        return []
    try:
        text = OUTCOME_LOG.read_text(encoding="utf-8-sig").strip()
        return json.loads(text) if text else []
    except Exception:
        return []


def run_backtest(
    decision_fn: Callable[[dict], dict],
    starting_balance: float = 100_000.0,
    limit: Optional[int] = None,
) -> dict:
    """
    decision_fn takes a synthetic market_data dict (derived from the outcome's
    indicators_snapshot) and returns a {action, confidence, ...} dict.

    For each historical outcome we keep the realised profit when the simulated
    decision matches the action actually taken (BUY/SELL). If the simulated
    decision is HOLD, that trade is skipped.
    """
    outcomes = _load_outcomes()
    if limit:
        outcomes = outcomes[-limit:]

    equity = starting_balance
    trades = 0
    wins = 0
    total_pnl = 0.0
    skipped = 0
    per_sym: dict[str, float] = {}

    for o in outcomes:
        snap = o.get("indicators_snapshot", {}) or {}
        market_data = {
            "symbol": o.get("symbol"),
            "timeframe": "1m",
            **snap,
        }
        try:
            decision = decision_fn(market_data) or {}
        except Exception:
            decision = {"action": "HOLD"}

        action = str(decision.get("action", "HOLD")).upper()
        if action == "HOLD":
            skipped += 1
            continue

        # If simulated action agrees with the action originally taken, credit
        # the realised profit; otherwise invert it (we'd have traded the other
        # side against the same move).
        realised = float(o.get("profit", 0) or 0)
        original = str(o.get("action", "")).upper()
        pnl = realised if action == original else -realised

        equity += pnl
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1
        per_sym[o["symbol"]] = per_sym.get(o["symbol"], 0.0) + pnl

    wr = round(wins / trades * 100, 2) if trades else 0.0
    return {
        "starting_balance": starting_balance,
        "ending_equity": round(equity, 2),
        "trades": trades,
        "skipped": skipped,
        "win_rate_pct": wr,
        "total_pnl": round(total_pnl, 2),
        "per_symbol_pnl": {k: round(v, 2) for k, v in per_sym.items()},
    }


def confidence_threshold_fn(threshold: float) -> Callable[[dict], dict]:
    """Simple baseline: rsi-mean-reversion above given confidence."""
    def fn(md: dict) -> dict:
        rsi = md.get("rsi") or 50
        if rsi < 30:
            return {"action": "BUY", "confidence": 0.7 if rsi < 25 else 0.55}
        if rsi > 70:
            return {"action": "SELL", "confidence": 0.7 if rsi > 75 else 0.55}
        return {"action": "HOLD", "confidence": 0.0}
    wrapped = fn
    if threshold > 0:
        def gated(md: dict) -> dict:
            d = wrapped(md)
            if d.get("confidence", 0) < threshold:
                return {"action": "HOLD"}
            return d
        return gated
    return wrapped
