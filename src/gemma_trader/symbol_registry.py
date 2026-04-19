"""
Symbol Registry
================
Maps a generic symbol name (e.g. "GOLD") to broker-specific tickers
(e.g. IC Markets: "XAUUSD", CFI: "XAUUSD_"). Persisted to symbols.yaml
so the settings UI can manage it without touching config.yaml.

Data shape (symbols.yaml):
    active_broker: ic_markets
    symbols:
      - generic: GOLD
        enabled: true
        active: true
        aliases:
          ic_markets: XAUUSD
          cfi: XAUUSD_
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "symbols.yaml"


@dataclass
class Symbol:
    generic: str
    enabled: bool = True
    active: bool = True
    aliases: dict = field(default_factory=dict)

    def resolve(self, broker: str) -> str:
        return self.aliases.get(broker, self.generic)


class SymbolRegistry:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.active_broker: str = "ic_markets"
        self.symbols: dict[str, Symbol] = {}
        self.load()

    # ── persistence ──
    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self.symbols = {}
                return
            try:
                data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except Exception as e:
                logger.error(f"Failed to load {self.path}: {e}")
                return
            self.active_broker = data.get("active_broker", "ic_markets")
            self.symbols = {}
            for entry in data.get("symbols", []):
                if "generic" not in entry:
                    continue
                s = Symbol(
                    generic=entry["generic"],
                    enabled=bool(entry.get("enabled", True)),
                    active=bool(entry.get("active", True)),
                    aliases=dict(entry.get("aliases") or {}),
                )
                self.symbols[s.generic] = s

    def save(self) -> None:
        with self._lock:
            data = {
                "active_broker": self.active_broker,
                "symbols": [asdict(s) for s in self.symbols.values()],
            }
            tmp = self.path.with_suffix(".yaml.tmp")
            tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            tmp.replace(self.path)

    # ── queries ──
    def resolve(self, generic: str, broker: Optional[str] = None) -> str:
        """Map generic → broker ticker. Unknown generic → return as-is."""
        broker = broker or self.active_broker
        sym = self.symbols.get(generic)
        return sym.resolve(broker) if sym else generic

    def reverse(self, broker_ticker: str, broker: Optional[str] = None) -> str:
        """Map broker ticker → generic. Unknown → return as-is."""
        broker = broker or self.active_broker
        for s in self.symbols.values():
            if s.aliases.get(broker) == broker_ticker or s.generic == broker_ticker:
                return s.generic
        return broker_ticker

    def active_generics(self) -> list[str]:
        return [s.generic for s in self.symbols.values() if s.enabled and s.active]

    def active_for_broker(self, broker: Optional[str] = None) -> list[str]:
        broker = broker or self.active_broker
        return [s.resolve(broker) for s in self.symbols.values() if s.enabled and s.active]

    def list(self) -> list[dict]:
        return [asdict(s) for s in self.symbols.values()]

    # ── mutations ──
    def upsert(self, generic: str, aliases: Optional[dict] = None,
               enabled: bool = True, active: bool = True) -> Symbol:
        generic = generic.strip()
        if not generic:
            raise ValueError("generic name required")
        existing = self.symbols.get(generic)
        if existing:
            if aliases is not None:
                existing.aliases = dict(aliases)
            existing.enabled = bool(enabled)
            existing.active = bool(active)
        else:
            existing = Symbol(
                generic=generic,
                enabled=bool(enabled),
                active=bool(active),
                aliases=dict(aliases or {}),
            )
            self.symbols[generic] = existing
        self.save()
        return existing

    def remove(self, generic: str) -> bool:
        if generic in self.symbols:
            del self.symbols[generic]
            self.save()
            return True
        return False

    def set_enabled(self, generic: str, enabled: bool) -> bool:
        s = self.symbols.get(generic)
        if not s:
            return False
        s.enabled = bool(enabled)
        self.save()
        return True

    def set_active(self, generic: str, active: bool) -> bool:
        s = self.symbols.get(generic)
        if not s:
            return False
        s.active = bool(active)
        self.save()
        return True

    def set_active_broker(self, broker: str) -> None:
        self.active_broker = broker.strip()
        self.save()


_singleton: Optional[SymbolRegistry] = None


def get_registry() -> SymbolRegistry:
    global _singleton
    if _singleton is None:
        _singleton = SymbolRegistry()
    return _singleton


