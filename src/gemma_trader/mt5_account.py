"""
MT5 Account Manager
====================
Single-account MT5 configuration with live connection testing.

Persists account credentials to `mt5_account.yaml` (git-ignored).
Provides magic-filtered position/deal queries so the bot only sees
trades it owns (magic=240411), ignoring anything placed by the user
or other strategies on the same account.

Usage:
    from gemma_trader.mt5_account import get_account

    acct = get_account()
    if not acct.is_configured():
        # User must enter credentials via /settings
        pass

    result = acct.test_connection()
    if result["ok"]:
        symbols = acct.list_symbols()  # broker's full list
        info = acct.get_info()          # balance, equity, leverage
        positions = acct.get_own_positions()  # magic-filtered
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

MAGIC = 240411

DEFAULT_ACCOUNT_PATH = Path("mt5_account.yaml")


class MT5Account:
    """Singleton managing a single MT5 account connection."""

    def __init__(self, account_path: Path = DEFAULT_ACCOUNT_PATH):
        self.account_path = Path(account_path)
        self._lock = threading.Lock()
        self._config = {
            "login": 0,
            "password": "",
            "server": "",
            "path": "",
        }
        self._mt5 = None
        self._connected = False
        self.load()

    # ── Persistence ──

    def load(self) -> None:
        """Load account config from YAML if present."""
        if self.account_path.exists():
            try:
                data = yaml.safe_load(self.account_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._config.update({
                        "login": int(data.get("login", 0) or 0),
                        "password": str(data.get("password", "") or ""),
                        "server": str(data.get("server", "") or ""),
                        "path": str(data.get("path", "") or ""),
                    })
            except Exception as e:
                logger.warning(f"Failed to load {self.account_path}: {e}")

    def save(
        self,
        login: int,
        password: str,
        server: str,
        path: str = "",
    ) -> None:
        """Persist account config. Restricts file permissions to 0600 on Unix."""
        with self._lock:
            self._config = {
                "login": int(login),
                "password": str(password),
                "server": str(server),
                "path": str(path or ""),
            }
            self.account_path.parent.mkdir(parents=True, exist_ok=True)
            self.account_path.write_text(
                yaml.safe_dump(self._config, default_flow_style=False),
                encoding="utf-8",
            )
            # Best-effort permission restriction on Unix (no-op on Windows)
            try:
                os.chmod(self.account_path, 0o600)
            except OSError:
                pass
        # Force reconnect on next use
        self.disconnect()

    def is_configured(self) -> bool:
        return bool(self._config["login"]) and bool(self._config["server"])

    def get_masked_config(self) -> dict:
        """Return config with password masked — safe for API responses."""
        return {
            "login": self._config["login"],
            "password": "***" if self._config["password"] else "",
            "server": self._config["server"],
            "path": self._config["path"],
            "configured": self.is_configured(),
            "has_password": bool(self._config["password"]),
        }

    # ── Connection ──

    def _import_mt5(self):
        """Import MetaTrader5 module; return None on failure."""
        if self._mt5 is not None:
            return self._mt5
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            return mt5
        except ImportError as e:
            import sys
            logger.warning(
                f"MetaTrader5 package not installed in {sys.executable}. "
                f"Install it with: {sys.executable} -m pip install MetaTrader5"
            )
            self._import_error = (
                f"Not installed in {sys.executable}. "
                f"Run: {sys.executable} -m pip install MetaTrader5"
            )
            return None

    def connect(self) -> tuple[bool, str]:
        """Initialize MT5 and log in. Returns (ok, error_message)."""
        mt5 = self._import_mt5()
        if mt5 is None:
            detail = getattr(self, "_import_error", "MetaTrader5 package not installed")
            return False, f"MetaTrader5 package not installed — {detail}"

        if not self.is_configured():
            return False, "Account not configured"

        try:
            init_kwargs = {}
            if self._config["path"]:
                init_kwargs["path"] = self._config["path"]

            if not mt5.initialize(**init_kwargs):
                err = mt5.last_error() if hasattr(mt5, "last_error") else "unknown"
                return False, f"MT5 initialize failed: {err}"

            authorized = mt5.login(
                login=self._config["login"],
                password=self._config["password"],
                server=self._config["server"],
            )
            if not authorized:
                err = mt5.last_error() if hasattr(mt5, "last_error") else "unknown"
                return False, f"MT5 login failed: {err}"

            self._connected = True
            return True, ""
        except Exception as e:
            return False, f"MT5 connection error: {e}"

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if self._mt5 is not None and self._connected:
            try:
                self._mt5.shutdown()
            except Exception:
                pass
        self._connected = False

    def test_connection(self) -> dict:
        """
        Attempt connection and return diagnostic info.
        Safe to call repeatedly — does not persist state changes.
        """
        ok, error = self.connect()
        if not ok:
            return {"ok": False, "error": error}

        try:
            info = self._mt5.account_info()
            if info is None:
                return {"ok": False, "error": "account_info returned None"}
            return {
                "ok": True,
                "balance": float(info.balance),
                "equity": float(info.equity),
                "margin": float(info.margin),
                "margin_free": float(info.margin_free),
                "leverage": int(info.leverage),
                "currency": str(info.currency),
                "server": str(info.server),
                "login": int(info.login),
                "name": str(getattr(info, "name", "")),
            }
        except Exception as e:
            return {"ok": False, "error": f"account_info error: {e}"}

    # ── Account info ──

    def get_info(self) -> Optional[dict]:
        """Return a fresh account_info snapshot, or None if unreachable."""
        if not self._connected:
            ok, _ = self.connect()
            if not ok:
                return None
        try:
            info = self._mt5.account_info()
            if info is None:
                return None
            return {
                "balance": float(info.balance),
                "equity": float(info.equity),
                "margin": float(info.margin),
                "margin_free": float(info.margin_free),
                "margin_level": float(getattr(info, "margin_level", 0) or 0),
                "leverage": int(info.leverage),
                "currency": str(info.currency),
                "server": str(info.server),
                "login": int(info.login),
                "name": str(getattr(info, "name", "")),
            }
        except Exception as e:
            logger.warning(f"get_info failed: {e}")
            return None

    # ── Symbol catalogue ──

    def list_symbols(self, group: Optional[str] = None) -> list[dict]:
        """
        Return full broker symbol list with min/max/step lot metadata.

        Args:
            group: Optional MT5 group filter (e.g. "*USD*").

        Returns: list of dicts with name, description, min_lot, max_lot,
                 step, digits, contract_size, currency_profit.
        """
        if not self._connected:
            ok, _ = self.connect()
            if not ok:
                return []
        try:
            symbols = (
                self._mt5.symbols_get(group)
                if group
                else self._mt5.symbols_get()
            )
            if not symbols:
                return []
            return [self._symbol_to_dict(s) for s in symbols]
        except Exception as e:
            logger.warning(f"list_symbols failed: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get metadata for a single symbol (min/max/step lot, digits, etc.)."""
        if not self._connected:
            ok, _ = self.connect()
            if not ok:
                return None
        try:
            s = self._mt5.symbol_info(symbol)
            if s is None:
                return None
            return self._symbol_to_dict(s)
        except Exception as e:
            logger.warning(f"get_symbol_info({symbol}) failed: {e}")
            return None

    @staticmethod
    def _symbol_to_dict(s) -> dict:
        """Convert an MT5 SymbolInfo object to a plain dict."""
        return {
            "name": str(getattr(s, "name", "")),
            "description": str(getattr(s, "description", "")),
            "min_lot": float(getattr(s, "volume_min", 0.01)),
            "max_lot": float(getattr(s, "volume_max", 100.0)),
            "step": float(getattr(s, "volume_step", 0.01)),
            "digits": int(getattr(s, "digits", 2)),
            "contract_size": float(getattr(s, "trade_contract_size", 1)),
            "currency_profit": str(getattr(s, "currency_profit", "")),
            "currency_base": str(getattr(s, "currency_base", "")),
            "spread": int(getattr(s, "spread", 0)),
            "trade_mode": int(getattr(s, "trade_mode", 0)),
            "visible": bool(getattr(s, "visible", True)),
        }

    # ── Position/Deal filtering (MAGIC-ONLY) ──

    def get_own_positions(self) -> list[dict]:
        """
        Return positions owned by this bot (magic=240411).
        Everything else on the account is ignored.
        """
        if not self._connected:
            ok, _ = self.connect()
            if not ok:
                return []
        try:
            all_positions = self._mt5.positions_get() or []
            result = []
            for p in all_positions:
                if int(getattr(p, "magic", 0)) != MAGIC:
                    continue
                result.append({
                    "ticket": int(p.ticket),
                    "symbol": str(p.symbol),
                    "type": int(p.type),  # 0=BUY, 1=SELL
                    "volume": float(p.volume),
                    "price_open": float(p.price_open),
                    "price_current": float(p.price_current),
                    "sl": float(p.sl),
                    "tp": float(p.tp),
                    "profit": float(p.profit),
                    "swap": float(p.swap),
                    "time": int(p.time),
                    "magic": int(p.magic),
                    "comment": str(getattr(p, "comment", "")),
                })
            return result
        except Exception as e:
            logger.warning(f"get_own_positions failed: {e}")
            return []

    def get_own_deals(self, days: int = 30) -> list[dict]:
        """Return deal history filtered to bot's magic."""
        if not self._connected:
            ok, _ = self.connect()
            if not ok:
                return []
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            deals = self._mt5.history_deals_get(start, end) or []
            result = []
            for d in deals:
                if int(getattr(d, "magic", 0)) != MAGIC:
                    continue
                result.append({
                    "ticket": int(d.ticket),
                    "order": int(getattr(d, "order", 0)),
                    "time": int(d.time),
                    "type": int(d.type),
                    "symbol": str(d.symbol),
                    "volume": float(d.volume),
                    "price": float(d.price),
                    "profit": float(d.profit),
                    "commission": float(getattr(d, "commission", 0)),
                    "swap": float(getattr(d, "swap", 0)),
                    "magic": int(d.magic),
                })
            return result
        except Exception as e:
            logger.warning(f"get_own_deals failed: {e}")
            return []

    def count_own_open_trades(self) -> int:
        return len(self.get_own_positions())


# ── Singleton ──

_account: Optional[MT5Account] = None


def get_account(path: Path = DEFAULT_ACCOUNT_PATH) -> MT5Account:
    global _account
    if _account is None:
        _account = MT5Account(path)
    return _account


def reset_account() -> None:
    """Reset singleton (mainly for tests)."""
    global _account
    if _account is not None:
        _account.disconnect()
    _account = None
