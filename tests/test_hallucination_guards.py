"""Tests for LLM hallucination guards in gemma_analyzer."""

import json
import math

import pytest

from gemma_trader.gemma_analyzer import (
    _detect_hallucinations,
    _validate_decision,
    _log_hallucination,
)


def test_symbol_mismatch_detected():
    decision = {"symbol": "ETHUSD", "action": "BUY", "confidence": 0.8,
                "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}
    market = {"symbol": "BTCUSD"}
    flags = _detect_hallucinations(decision, market)
    assert "symbol_mismatch" in flags


def test_unknown_symbol_detected():
    decision = {"symbol": "DOGECOIN", "action": "BUY", "confidence": 0.8,
                "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}
    flags = _detect_hallucinations(
        decision,
        market_data=None,
        allowed_symbols=["BTCUSD", "ETHUSD"],
    )
    assert "unknown_symbol" in flags


def test_nan_detected():
    decision = {"action": "BUY", "confidence": float("nan"),
                "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}
    flags = _detect_hallucinations(decision)
    assert "nan_value" in flags


def test_inf_detected():
    decision = {"action": "BUY", "confidence": 0.8,
                "sl_distance_atr": float("inf"), "tp_distance_atr": 1.5}
    flags = _detect_hallucinations(decision)
    assert "nan_value" in flags


def test_reason_too_long_detected():
    decision = {"action": "HOLD", "confidence": 0.0,
                "reason": "x" * 600}
    flags = _detect_hallucinations(decision)
    assert "reason_too_long" in flags


def test_low_confidence_trade_detected():
    decision = {"action": "BUY", "confidence": 0.15,
                "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}
    flags = _detect_hallucinations(decision)
    assert "low_confidence_trade" in flags


def test_inverted_tp_sl_detected():
    decision = {"action": "BUY", "confidence": 0.8,
                "sl_distance_atr": 2.0, "tp_distance_atr": 0.5}
    flags = _detect_hallucinations(decision)
    assert "inverted_tp_sl" in flags


def test_out_of_range_sl_detected():
    decision = {"action": "BUY", "confidence": 0.8,
                "sl_distance_atr": 50.0, "tp_distance_atr": 1.5}
    flags = _detect_hallucinations(decision)
    assert "out_of_range_sl_tp" in flags


def test_invalid_action_detected():
    decision = {"action": "LONG", "confidence": 0.8,
                "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}
    flags = _detect_hallucinations(decision)
    assert "invalid_action" in flags


def test_clean_decision_no_flags():
    decision = {"symbol": "BTCUSD", "action": "BUY", "confidence": 0.8,
                "sl_distance_atr": 1.0, "tp_distance_atr": 1.5,
                "reason": "strong signal"}
    market = {"symbol": "BTCUSD"}
    flags = _detect_hallucinations(decision, market, ["BTCUSD"])
    assert flags == []


def test_validate_decision_forces_hold_on_nan(monkeypatch, tmp_path):
    """NaN confidence must become HOLD, never leak into a trade."""
    monkeypatch.setattr(
        "gemma_trader.gemma_analyzer.HALLUCINATION_LOG",
        tmp_path / "hallucinations.json",
    )
    decision = {
        "symbol": "BTCUSD", "action": "BUY",
        "confidence": float("nan"),
        "sl_distance_atr": 1.0, "tp_distance_atr": 1.5,
    }
    validated = _validate_decision(decision, market_data={"symbol": "BTCUSD"})
    assert validated["action"] == "HOLD"
    assert validated["confidence"] == 0.0
    assert "hallucination guard" in validated["reason"]


def test_validate_decision_fixes_symbol_mismatch(monkeypatch, tmp_path):
    """Symbol mismatch is corrected to the requested symbol."""
    monkeypatch.setattr(
        "gemma_trader.gemma_analyzer.HALLUCINATION_LOG",
        tmp_path / "h.json",
    )
    decision = {"symbol": "ETHUSD", "action": "BUY", "confidence": 0.8,
                "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}
    validated = _validate_decision(decision, market_data={"symbol": "BTCUSD"})
    assert validated["symbol"] == "BTCUSD"
    assert "hallucination_flags" in validated
    assert "symbol_mismatch" in validated["hallucination_flags"]


def test_validate_decision_truncates_long_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "gemma_trader.gemma_analyzer.HALLUCINATION_LOG",
        tmp_path / "h.json",
    )
    decision = {"symbol": "BTCUSD", "action": "HOLD", "confidence": 0.0,
                "reason": "y" * 700}
    validated = _validate_decision(decision)
    assert len(validated["reason"]) <= 500
    assert validated["reason"].endswith("...")


def test_log_hallucination_writes_json(tmp_path, monkeypatch):
    log_path = tmp_path / "h.json"
    monkeypatch.setattr("gemma_trader.gemma_analyzer.HALLUCINATION_LOG", log_path)

    _log_hallucination(
        ["symbol_mismatch"],
        {"action": "BUY", "symbol": "ETHUSD", "confidence": 0.8},
        {"symbol": "BTCUSD"},
    )
    assert log_path.exists()
    data = json.loads(log_path.read_text())
    assert len(data) == 1
    assert data[0]["flags"] == ["symbol_mismatch"]
    assert data[0]["symbol_requested"] == "BTCUSD"
    assert data[0]["symbol_returned"] == "ETHUSD"


def test_log_hallucination_rolls_at_500(tmp_path, monkeypatch):
    log_path = tmp_path / "h.json"
    monkeypatch.setattr("gemma_trader.gemma_analyzer.HALLUCINATION_LOG", log_path)

    # Write 500 pre-existing entries
    initial = [{"timestamp": "t", "flags": ["old"]} for _ in range(500)]
    log_path.write_text(json.dumps(initial))

    _log_hallucination(["new_flag"], {"action": "HOLD"}, None)
    data = json.loads(log_path.read_text())
    # Still 500 after one append — oldest dropped
    assert len(data) == 500
    assert data[-1]["flags"] == ["new_flag"]


def test_log_hallucination_noop_on_empty_flags(tmp_path, monkeypatch):
    log_path = tmp_path / "h.json"
    monkeypatch.setattr("gemma_trader.gemma_analyzer.HALLUCINATION_LOG", log_path)
    _log_hallucination([], {"action": "BUY"}, None)
    assert not log_path.exists()
