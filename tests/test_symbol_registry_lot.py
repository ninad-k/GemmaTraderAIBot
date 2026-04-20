"""Tests for lot_size extension of SymbolRegistry."""

import yaml

from gemma_trader.symbol_registry import SymbolRegistry


def test_lot_size_default(tmp_path):
    path = tmp_path / "symbols.yaml"
    reg = SymbolRegistry(path=path)
    reg.upsert("BTCUSD", aliases={"mt5": "BTCUSD"})
    assert reg.get_lot_size("BTCUSD") == 0.01


def test_lot_size_persisted(tmp_path):
    path = tmp_path / "symbols.yaml"
    reg = SymbolRegistry(path=path)
    reg.upsert("BTCUSD", aliases={"mt5": "BTCUSD"}, lot_size=0.5)

    # Reload from disk
    reg2 = SymbolRegistry(path=path)
    assert reg2.get_lot_size("BTCUSD") == 0.5


def test_set_lot_size_updates(tmp_path):
    path = tmp_path / "symbols.yaml"
    reg = SymbolRegistry(path=path)
    reg.upsert("ETHUSD", aliases={"mt5": "ETHUSD"})
    ok = reg.set_lot_size("ETHUSD", 0.25)
    assert ok
    assert reg.get_lot_size("ETHUSD") == 0.25


def test_set_lot_size_unknown_symbol(tmp_path):
    reg = SymbolRegistry(path=tmp_path / "symbols.yaml")
    assert reg.set_lot_size("NONEXISTENT", 1.0) is False


def test_backward_compat_missing_lot_field(tmp_path):
    """Old symbols.yaml without lot_size should load with default 0.01."""
    path = tmp_path / "symbols.yaml"
    path.write_text(yaml.safe_dump({
        "active_broker": "mt5",
        "symbols": [
            {"generic": "BTCUSD", "enabled": True, "active": True,
             "aliases": {"mt5": "BTCUSD"}},
        ],
    }))
    reg = SymbolRegistry(path=path)
    assert reg.get_lot_size("BTCUSD") == 0.01


def test_upsert_updates_lot_size(tmp_path):
    path = tmp_path / "symbols.yaml"
    reg = SymbolRegistry(path=path)
    reg.upsert("SOLUSD", lot_size=0.1)
    reg.upsert("SOLUSD", lot_size=0.3)
    assert reg.get_lot_size("SOLUSD") == 0.3
