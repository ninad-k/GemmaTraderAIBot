"""
Multi-channel notifier: Telegram, Microsoft Teams, WhatsApp (Meta Cloud API).

Config lives in notifications.yaml (git-ignored), example:

    channels:
      telegram:
        enabled: true
        bot_token: "..."
        chat_id: "..."
      teams:
        enabled: true
        webhook_url: "https://outlook.office.com/webhook/..."
      whatsapp:
        enabled: true
        phone_number_id: "..."
        access_token: "..."
        to: "+15555551234"
    events:
      entry: true
      exit: true
      halt: true
      breaker: true
      reconnect: true
      resume: true
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import requests
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "notifications.yaml"

DEFAULT_CONFIG = {
    "channels": {
        "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
        "teams": {"enabled": False, "webhook_url": ""},
        "whatsapp": {
            "enabled": False, "phone_number_id": "",
            "access_token": "", "to": "",
        },
    },
    "events": {
        "entry": True, "exit": True, "halt": True,
        "breaker": True, "reconnect": True, "resume": True,
    },
}


class Notifier:
    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.config = dict(DEFAULT_CONFIG)
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            try:
                data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
                # shallow-merge with defaults
                channels = {**DEFAULT_CONFIG["channels"], **(data.get("channels") or {})}
                events = {**DEFAULT_CONFIG["events"], **(data.get("events") or {})}
                self.config = {"channels": channels, "events": events}
            except Exception as e:
                logger.error(f"notifier config load failed: {e}")

    def save(self, new_config: dict) -> None:
        with self._lock:
            self.config = new_config
            tmp = self.path.with_suffix(".yaml.tmp")
            tmp.write_text(yaml.safe_dump(new_config, sort_keys=False), encoding="utf-8")
            tmp.replace(self.path)

    def get_config(self) -> dict:
        return self.config

    # ── dispatch ──
    def notify(self, event: str, message: str, meta: Optional[dict] = None) -> dict:
        if not self.config["events"].get(event, False):
            return {"skipped": f"event '{event}' disabled"}

        text = f"[Rey Capital] {event.upper()}: {message}"
        if meta:
            pairs = ", ".join(f"{k}={v}" for k, v in meta.items())
            text = f"{text} | {pairs}"

        results = {}
        channels = self.config["channels"]
        if channels["telegram"].get("enabled"):
            results["telegram"] = self._send_telegram(text, channels["telegram"])
        if channels["teams"].get("enabled"):
            results["teams"] = self._send_teams(text, channels["teams"])
        if channels["whatsapp"].get("enabled"):
            results["whatsapp"] = self._send_whatsapp(text, channels["whatsapp"])
        return results

    # ── transports ──
    @staticmethod
    def _send_telegram(text: str, cfg: dict) -> dict:
        token = cfg.get("bot_token")
        chat = cfg.get("chat_id")
        if not token or not chat:
            return {"error": "missing bot_token or chat_id"}
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": text},
                timeout=8,
            )
            return {"status": r.status_code}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _send_teams(text: str, cfg: dict) -> dict:
        url = cfg.get("webhook_url")
        if not url:
            return {"error": "missing webhook_url"}
        try:
            r = requests.post(url, json={"text": text}, timeout=8)
            return {"status": r.status_code}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _send_whatsapp(text: str, cfg: dict) -> dict:
        phone_id = cfg.get("phone_number_id")
        token = cfg.get("access_token")
        to = cfg.get("to")
        if not (phone_id and token and to):
            return {"error": "missing whatsapp credentials"}
        try:
            r = requests.post(
                f"https://graph.facebook.com/v20.0/{phone_id}/messages",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "text",
                    "text": {"body": text},
                },
                timeout=8,
            )
            return {"status": r.status_code}
        except Exception as e:
            return {"error": str(e)}


_singleton: Optional[Notifier] = None


def get_notifier() -> Notifier:
    global _singleton
    if _singleton is None:
        _singleton = Notifier()
    return _singleton


