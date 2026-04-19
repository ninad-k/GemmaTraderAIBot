from datetime import datetime, timedelta
from pathlib import Path

import pytest

from risk_manager import RiskManager


BASE_CONFIG = {
    "trading": {
        "mode": "paper",
        "confidence_threshold": 0.6,
        "max_position_size_pct": 1.0,
        "max_open_trades": 2,
        "cooldown_minutes": 3,
        "allowed_symbols": ["BTCUSD", "ETHUSD"],
    },
    "risk_management": {
        "stop_loss_atr_multiplier": 1.0,
        "take_profit_atr_multiplier": 1.5,
        "max_daily_loss_pct": 5.0,
        "max_drawdown_pct": 10.0,
    },
    "adaptive": {
        "enabled": True,
        "min_trades_for_adaptation": 1,
        "cooldown_on_streak_loss": 2,
        "cooldown_duration_minutes": 15,
        "max_confidence_threshold": 0.85,
        "min_confidence_threshold": 0.5,
    },
    "logging": {},
}


def _rm(tmp_path: Path) -> RiskManager:
    cfg = {**BASE_CONFIG,
           "logging": {
               "trade_log": str(tmp_path / "trades.json"),
               "outcome_log": str(tmp_path / "outcomes.json"),
               "parameter_adjustments": str(tmp_path / "adj.json"),
           }}
    return RiskManager(cfg)


def test_rejects_unknown_symbol(tmp_path):
    rm = _rm(tmp_path)
    ok, _ = rm.can_trade({"symbol": "FOO", "confidence": 0.9}, {})
    assert ok is False


def test_rejects_low_confidence(tmp_path):
    rm = _rm(tmp_path)
    ok, _ = rm.can_trade({"symbol": "BTCUSD", "confidence": 0.3}, {})
    assert ok is False


def test_respects_max_open_trades(tmp_path):
    rm = _rm(tmp_path)
    rm.open_trades = [{"symbol": "X1"}, {"symbol": "X2"}]
    ok, _ = rm.can_trade({"symbol": "BTCUSD", "confidence": 0.9}, {})
    assert ok is False


def test_loss_streak_triggers_cooldown(tmp_path):
    rm = _rm(tmp_path)
    rm._update_streak("BTCUSD", False)
    rm._update_streak("BTCUSD", False)  # 2 losses = cooldown
    assert "BTCUSD" in rm.cooled_down_symbols


def test_threshold_raises_on_low_winrate(tmp_path):
    rm = _rm(tmp_path)
    start = rm.current_threshold
    rm.adjust_threshold(win_rate=20.0, total_trades=10)
    assert rm.current_threshold > start


def test_threshold_lowers_on_high_winrate(tmp_path):
    rm = _rm(tmp_path)
    rm.current_threshold = 0.8
    rm.adjust_threshold(win_rate=80.0, total_trades=10)
    assert rm.current_threshold < 0.8
