"""
Safety subsystem: kill-switch, equity-curve circuit breaker, heartbeat watchdog.

State is kept in-memory + mirrored to logs/safety_state.json so it survives
restarts. The trader loop queries `is_halted()` each cycle and refuses new
entries when tripped.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_PATH = Path("logs/safety_state.json")


@dataclass
class SafetyState:
    halted: bool = False
    halt_reason: str = ""
    halted_at: str = ""
    peak_equity: float = 0.0
    last_equity: float = 0.0
    last_heartbeat: str = ""
    breaker_tripped: bool = False
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class SafetyController:
    def __init__(self, max_drawdown_pct: float = 10.0):
        self.max_drawdown_pct = float(max_drawdown_pct)
        self._lock = threading.Lock()
        self.state = SafetyState()
        self._notifier = None
        self._load()

    def attach_notifier(self, notifier) -> None:
        self._notifier = notifier

    # ── persistence ──
    def _load(self) -> None:
        if STATE_PATH.exists():
            try:
                self.state = SafetyState(**json.loads(STATE_PATH.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(f"safety state corrupt, resetting: {e}")

    def _save(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self.state.to_dict(), indent=2))

    # ── kill switch ──
    def halt(self, reason: str, source: str = "manual") -> None:
        with self._lock:
            if self.state.halted:
                return
            self.state.halted = True
            self.state.halt_reason = reason
            self.state.halted_at = datetime.now().isoformat()
            self.state.history.append({
                "event": "halt", "reason": reason, "source": source,
                "timestamp": self.state.halted_at,
            })
            self._save()
        logger.warning(f"[SAFETY] HALT ({source}): {reason}")
        if self._notifier:
            try:
                self._notifier.notify("halt", f"Trading halted: {reason}", {"source": source})
            except Exception as e:
                logger.warning(f"notify failed: {e}")

    def resume(self) -> None:
        with self._lock:
            if not self.state.halted:
                return
            self.state.halted = False
            self.state.halt_reason = ""
            self.state.breaker_tripped = False
            self.state.history.append({
                "event": "resume", "timestamp": datetime.now().isoformat(),
            })
            self._save()
        logger.info("[SAFETY] Trading resumed")
        if self._notifier:
            try:
                self._notifier.notify("resume", "Trading resumed")
            except Exception:
                pass

    def is_halted(self) -> bool:
        return self.state.halted

    # ── equity curve circuit breaker ──
    def update_equity(self, equity: float) -> None:
        with self._lock:
            self.state.last_equity = float(equity)
            if equity > self.state.peak_equity:
                self.state.peak_equity = float(equity)
            peak = self.state.peak_equity or equity
            dd_pct = 0.0 if peak <= 0 else (peak - equity) / peak * 100
            self._save()
        if dd_pct >= self.max_drawdown_pct and not self.state.breaker_tripped:
            with self._lock:
                self.state.breaker_tripped = True
            self.halt(
                f"Drawdown {dd_pct:.2f}% from peak {peak:.2f} exceeds {self.max_drawdown_pct}%",
                source="drawdown_breaker",
            )

    def drawdown_pct(self) -> float:
        peak = self.state.peak_equity
        if peak <= 0:
            return 0.0
        return (peak - self.state.last_equity) / peak * 100

    # ── heartbeat ──
    def heartbeat(self) -> None:
        with self._lock:
            self.state.last_heartbeat = datetime.now().isoformat()
            self._save()

    def seconds_since_heartbeat(self) -> float:
        if not self.state.last_heartbeat:
            return 0.0
        try:
            return (datetime.now() - datetime.fromisoformat(self.state.last_heartbeat)).total_seconds()
        except Exception:
            return 0.0


_singleton: Optional[SafetyController] = None


def get_safety(config: Optional[dict] = None) -> SafetyController:
    global _singleton
    if _singleton is None:
        max_dd = 10.0
        if config:
            max_dd = float(config.get("risk_management", {}).get("max_drawdown_pct", 10.0))
        _singleton = SafetyController(max_drawdown_pct=max_dd)
    return _singleton


def flatten_all_positions(broker, feed=None) -> dict:
    """Best-effort close of all bot-owned positions."""
    closed = []
    errors = []
    try:
        if feed is not None:
            positions = feed.get_positions()
            for p in positions:
                if p.get("magic") != 240411:
                    continue
                try:
                    r = broker.close_position(p["symbol"])
                    closed.append({"symbol": p["symbol"], "result": r})
                except Exception as e:
                    errors.append({"symbol": p["symbol"], "error": str(e)})
    except Exception as e:
        errors.append({"error": str(e)})
    return {"closed": closed, "errors": errors}


