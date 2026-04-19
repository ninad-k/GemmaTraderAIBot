"""
News blackout windows.

Simple config-driven list of UTC windows (start/end ISO timestamps) during
which the trader refuses new entries. Exits & position management continue
normally.

Config lives in news_blackouts.yaml:
    windows:
      - start: "2026-04-19T14:00:00"
        end:   "2026-04-19T14:30:00"
        label: "FOMC minutes"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "news_blackouts.yaml"


class NewsCalendar:
    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self.windows: list[dict] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.windows = []
            return
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            self.windows = list(data.get("windows") or [])
        except Exception as e:
            logger.error(f"news calendar load failed: {e}")
            self.windows = []

    def save(self) -> None:
        self.path.write_text(
            yaml.safe_dump({"windows": self.windows}, sort_keys=False),
            encoding="utf-8",
        )

    def in_blackout(self, now: Optional[datetime] = None) -> tuple[bool, str]:
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)
        for w in self.windows:
            try:
                s = datetime.fromisoformat(str(w["start"]))
                e = datetime.fromisoformat(str(w["end"]))
            except Exception:
                continue
            if s <= now <= e:
                return True, str(w.get("label", "news blackout"))
        return False, ""

    def add(self, start: str, end: str, label: str = "") -> None:
        self.windows.append({"start": start, "end": end, "label": label})
        self.save()

    def remove(self, index: int) -> bool:
        if 0 <= index < len(self.windows):
            self.windows.pop(index)
            self.save()
            return True
        return False


_singleton: Optional[NewsCalendar] = None


def get_calendar() -> NewsCalendar:
    global _singleton
    if _singleton is None:
        _singleton = NewsCalendar()
    return _singleton


