"""
Performance metrics: Sharpe, Sortino, MaxDD, equity curve,
per-regime + per-symbol attribution.

All computations read from logs/trade_outcomes.json so they're
cheap to regenerate on demand from the dashboard.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

OUTCOME_LOG = Path("logs/trade_outcomes.json")


def _load_outcomes() -> list[dict]:
    if not OUTCOME_LOG.exists():
        return []
    try:
        text = OUTCOME_LOG.read_text(encoding="utf-8-sig").strip()
        return json.loads(text) if text else []
    except Exception:
        return []


def equity_curve(start_balance: float = 100_000.0) -> list[dict]:
    outcomes = _load_outcomes()
    curve = []
    eq = start_balance
    peak = eq
    for o in outcomes:
        eq += float(o.get("profit", 0) or 0)
        peak = max(peak, eq)
        dd = 0.0 if peak <= 0 else (peak - eq) / peak * 100
        curve.append({
            "time": o.get("close_time"),
            "equity": round(eq, 2),
            "drawdown_pct": round(dd, 3),
            "symbol": o.get("symbol"),
            "result": o.get("result"),
            "profit": o.get("profit"),
        })
    return curve


def _returns(outcomes: Iterable[dict]) -> list[float]:
    return [float(o.get("profit", 0) or 0) for o in outcomes]


def sharpe(returns: list[float], rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    avg = mean(returns) - rf
    sd = pstdev(returns)
    if sd == 0:
        return 0.0
    return round(avg / sd * math.sqrt(252), 3)


def sortino(returns: list[float], rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    avg = mean(returns) - rf
    downside = [r for r in returns if r < 0]
    if not downside:
        return 0.0
    dd_sd = pstdev(downside) if len(downside) > 1 else abs(downside[0])
    if dd_sd == 0:
        return 0.0
    return round(avg / dd_sd * math.sqrt(252), 3)


def max_drawdown(start_balance: float = 100_000.0) -> float:
    curve = equity_curve(start_balance)
    if not curve:
        return 0.0
    return round(max(p["drawdown_pct"] for p in curve), 3)


def summary(start_balance: float = 100_000.0) -> dict:
    outcomes = _load_outcomes()
    rets = _returns(outcomes)
    wins = sum(1 for r in rets if r > 0)
    losses = sum(1 for r in rets if r <= 0)
    total_pnl = sum(rets)
    wr = round(wins / len(rets) * 100, 2) if rets else 0
    avg_win = round(mean([r for r in rets if r > 0]), 2) if wins else 0
    avg_loss = round(mean([r for r in rets if r <= 0]), 2) if losses else 0
    profit_factor = 0.0
    gross_profit = sum(r for r in rets if r > 0)
    gross_loss = abs(sum(r for r in rets if r < 0))
    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 3)
    return {
        "trades": len(rets),
        "wins": wins, "losses": losses,
        "win_rate_pct": wr,
        "total_pnl": round(total_pnl, 2),
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "max_drawdown_pct": max_drawdown(start_balance),
    }


def per_symbol() -> dict:
    outcomes = _load_outcomes()
    bucket: dict[str, list[float]] = {}
    for o in outcomes:
        bucket.setdefault(o.get("symbol", "?"), []).append(float(o.get("profit", 0) or 0))
    out = {}
    for sym, rets in bucket.items():
        wins = sum(1 for r in rets if r > 0)
        out[sym] = {
            "trades": len(rets),
            "win_rate_pct": round(wins / len(rets) * 100, 2) if rets else 0,
            "pnl": round(sum(rets), 2),
            "sharpe": sharpe(rets),
        }
    return out


def per_regime() -> dict:
    """Bucket by indicators_snapshot.regime if present, else by vol_trend."""
    outcomes = _load_outcomes()
    bucket: dict[str, list[float]] = {}
    for o in outcomes:
        snap = o.get("indicators_snapshot", {}) or {}
        regime = snap.get("regime") or snap.get("vol_trend") or "UNKNOWN"
        bucket.setdefault(str(regime), []).append(float(o.get("profit", 0) or 0))
    out = {}
    for reg, rets in bucket.items():
        wins = sum(1 for r in rets if r > 0)
        out[reg] = {
            "trades": len(rets),
            "win_rate_pct": round(wins / len(rets) * 100, 2) if rets else 0,
            "pnl": round(sum(rets), 2),
        }
    return out
