"""Tests for MT5 account manager — mocked MT5 SDK, no real broker."""

import sys
from unittest.mock import MagicMock

import pytest
import yaml

from gemma_trader.mt5_account import MT5Account, MAGIC, reset_account


@pytest.fixture(autouse=True)
def clean_singleton():
    reset_account()
    yield
    reset_account()


def _mock_mt5_module(monkeypatch, *, account_info=None, positions=None,
                    symbols=None, deals=None, login_ok=True, init_ok=True):
    """Install a fake MetaTrader5 module into sys.modules."""
    mod = MagicMock()
    mod.initialize.return_value = init_ok
    mod.login.return_value = login_ok
    mod.shutdown.return_value = None
    mod.last_error.return_value = "mock_error"
    mod.account_info.return_value = account_info
    mod.positions_get.return_value = positions or []
    mod.symbols_get.return_value = symbols or []
    mod.symbol_info.return_value = None
    mod.history_deals_get.return_value = deals or []
    monkeypatch.setitem(sys.modules, "MetaTrader5", mod)
    return mod


def _mock_account_info(balance=10000.0, currency="USD", leverage=100):
    info = MagicMock()
    info.balance = balance
    info.equity = balance
    info.margin = 0.0
    info.margin_free = balance
    info.margin_level = 0.0
    info.leverage = leverage
    info.currency = currency
    info.server = "TestServer"
    info.login = 12345678
    info.name = "Test User"
    return info


def test_not_configured_initially(tmp_path):
    acct = MT5Account(account_path=tmp_path / "acct.yaml")
    assert not acct.is_configured()
    masked = acct.get_masked_config()
    assert masked["configured"] is False
    assert masked["password"] == ""


def test_save_persists_to_yaml(tmp_path):
    path = tmp_path / "acct.yaml"
    acct = MT5Account(account_path=path)
    acct.save(login=12345, password="secret", server="MyBroker-Demo", path="")

    assert path.exists()
    data = yaml.safe_load(path.read_text())
    assert data["login"] == 12345
    assert data["password"] == "secret"
    assert data["server"] == "MyBroker-Demo"


def test_load_preserves_values(tmp_path):
    path = tmp_path / "acct.yaml"
    path.write_text(yaml.safe_dump({
        "login": 999, "password": "abc", "server": "X", "path": "",
    }))
    acct = MT5Account(account_path=path)
    assert acct.is_configured()
    assert acct._config["login"] == 999
    assert acct._config["password"] == "abc"


def test_masked_config_hides_password(tmp_path):
    acct = MT5Account(account_path=tmp_path / "a.yaml")
    acct.save(12345, "mysecret", "SrvX")
    masked = acct.get_masked_config()
    assert masked["password"] == "***"
    assert masked["has_password"] is True
    assert "mysecret" not in str(masked)


def test_test_connection_success(tmp_path, monkeypatch):
    _mock_mt5_module(
        monkeypatch,
        account_info=_mock_account_info(balance=5000.0, currency="EUR", leverage=500),
    )
    acct = MT5Account(account_path=tmp_path / "a.yaml")
    acct.save(12345, "pwd", "Srv")

    result = acct.test_connection()
    assert result["ok"] is True
    assert result["balance"] == 5000.0
    assert result["currency"] == "EUR"
    assert result["leverage"] == 500


def test_test_connection_login_fails(tmp_path, monkeypatch):
    _mock_mt5_module(monkeypatch, login_ok=False)
    acct = MT5Account(account_path=tmp_path / "a.yaml")
    acct.save(12345, "wrong", "Srv")

    result = acct.test_connection()
    assert result["ok"] is False
    assert "login" in result["error"].lower()


def test_test_connection_not_configured(tmp_path, monkeypatch):
    _mock_mt5_module(monkeypatch)
    acct = MT5Account(account_path=tmp_path / "a.yaml")

    result = acct.test_connection()
    assert result["ok"] is False
    assert "not configured" in result["error"].lower()


def test_list_symbols_returns_metadata(tmp_path, monkeypatch):
    sym_a = MagicMock()
    sym_a.name = "BTCUSD"
    sym_a.description = "Bitcoin"
    sym_a.volume_min = 0.01
    sym_a.volume_max = 10.0
    sym_a.volume_step = 0.01
    sym_a.digits = 2
    sym_a.trade_contract_size = 1.0
    sym_a.currency_profit = "USD"
    sym_a.currency_base = "BTC"
    sym_a.spread = 5
    sym_a.trade_mode = 4
    sym_a.visible = True

    _mock_mt5_module(
        monkeypatch,
        account_info=_mock_account_info(),
        symbols=[sym_a],
    )
    acct = MT5Account(account_path=tmp_path / "a.yaml")
    acct.save(12345, "pwd", "Srv")

    symbols = acct.list_symbols()
    assert len(symbols) == 1
    assert symbols[0]["name"] == "BTCUSD"
    assert symbols[0]["min_lot"] == 0.01
    assert symbols[0]["description"] == "Bitcoin"


def test_get_own_positions_filters_by_magic(tmp_path, monkeypatch):
    """Foreign positions (different magic) must be excluded."""
    pos_own = MagicMock()
    pos_own.ticket = 1
    pos_own.symbol = "BTCUSD"
    pos_own.type = 0
    pos_own.volume = 0.1
    pos_own.price_open = 45000
    pos_own.price_current = 45500
    pos_own.sl = 44000
    pos_own.tp = 46000
    pos_own.profit = 50
    pos_own.swap = 0
    pos_own.time = 1700000000
    pos_own.magic = MAGIC  # ← bot's magic
    pos_own.comment = "gemma4"

    pos_foreign = MagicMock()
    pos_foreign.ticket = 2
    pos_foreign.symbol = "BTCUSD"
    pos_foreign.type = 0
    pos_foreign.volume = 0.5
    pos_foreign.price_open = 45000
    pos_foreign.price_current = 45100
    pos_foreign.sl = 0
    pos_foreign.tp = 0
    pos_foreign.profit = 10
    pos_foreign.swap = 0
    pos_foreign.time = 1700000000
    pos_foreign.magic = 999999  # ← foreign strategy
    pos_foreign.comment = "manual"

    _mock_mt5_module(
        monkeypatch,
        account_info=_mock_account_info(),
        positions=[pos_own, pos_foreign],
    )
    acct = MT5Account(account_path=tmp_path / "a.yaml")
    acct.save(12345, "pwd", "Srv")

    positions = acct.get_own_positions()
    assert len(positions) == 1
    assert positions[0]["ticket"] == 1
    assert positions[0]["magic"] == MAGIC


def test_get_own_deals_filters_by_magic(tmp_path, monkeypatch):
    d_own = MagicMock()
    d_own.ticket = 100
    d_own.order = 10
    d_own.time = 1700000000
    d_own.type = 0
    d_own.symbol = "BTCUSD"
    d_own.volume = 0.1
    d_own.price = 45000
    d_own.profit = 50
    d_own.commission = -0.5
    d_own.swap = 0
    d_own.magic = MAGIC

    d_foreign = MagicMock()
    d_foreign.ticket = 200
    d_foreign.order = 20
    d_foreign.time = 1700000000
    d_foreign.type = 0
    d_foreign.symbol = "BTCUSD"
    d_foreign.volume = 1.0
    d_foreign.price = 45000
    d_foreign.profit = -10
    d_foreign.commission = -1.0
    d_foreign.swap = 0
    d_foreign.magic = 999999

    _mock_mt5_module(
        monkeypatch,
        account_info=_mock_account_info(),
        deals=[d_own, d_foreign],
    )
    acct = MT5Account(account_path=tmp_path / "a.yaml")
    acct.save(12345, "pwd", "Srv")

    deals = acct.get_own_deals(days=30)
    assert len(deals) == 1
    assert deals[0]["ticket"] == 100


def test_disconnect_is_safe_without_connection(tmp_path):
    acct = MT5Account(account_path=tmp_path / "a.yaml")
    acct.disconnect()  # should not raise
    assert not acct._connected
