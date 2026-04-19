from pathlib import Path

from symbol_registry import SymbolRegistry


def test_resolve_and_reverse(tmp_path: Path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "active_broker: ic_markets\n"
        "symbols:\n"
        "  - generic: GOLD\n"
        "    enabled: true\n"
        "    active: true\n"
        "    aliases: {ic_markets: XAUUSD, cfi: XAUUSD_}\n",
        encoding="utf-8",
    )
    r = SymbolRegistry(path=p)
    assert r.resolve("GOLD") == "XAUUSD"
    assert r.resolve("GOLD", "cfi") == "XAUUSD_"
    assert r.reverse("XAUUSD_", "cfi") == "GOLD"
    # unknown generic passes through
    assert r.resolve("UNKNOWN") == "UNKNOWN"


def test_upsert_toggle_remove_persist(tmp_path: Path):
    p = tmp_path / "s.yaml"
    p.write_text("active_broker: ic_markets\nsymbols: []\n", encoding="utf-8")
    r = SymbolRegistry(path=p)
    r.upsert("BTCUSD", aliases={"ic_markets": "BTCUSD", "cfi": "BTCUSD_"})
    assert r.active_generics() == ["BTCUSD"]
    r.set_enabled("BTCUSD", False)
    assert r.active_generics() == []
    # roundtrip
    r2 = SymbolRegistry(path=p)
    assert "BTCUSD" in r2.symbols
    assert r2.symbols["BTCUSD"].enabled is False
    assert r.remove("BTCUSD") is True
    assert r.remove("BTCUSD") is False
