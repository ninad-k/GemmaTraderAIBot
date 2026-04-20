"""Tests for historical backtester."""

import pytest
import numpy as np
import pandas as pd

from gemma_trader.historical_backtester import HistoricalBacktester, BacktestResult
from gemma_trader.storage import reset_db, TradingDB


@pytest.fixture
def backtester(tmp_path, monkeypatch):
    reset_db()
    from gemma_trader import storage
    db = TradingDB(tmp_path / "test.db")
    monkeypatch.setattr(storage, "_db_instance", db)
    return HistoricalBacktester(cache_ohlcv=True)


@pytest.fixture
def synthetic_ohlcv():
    """Synthetic OHLCV with a clear trend for deterministic tests."""
    n = 500
    rng = np.random.default_rng(42)
    drift = 0.0005
    returns = rng.normal(drift, 0.002, n)
    close = 45000 * np.exp(returns.cumsum())
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": rng.uniform(1000, 5000, n),
    }, index=idx)


def test_always_hold_strategy_trades_zero(backtester, synthetic_ohlcv, monkeypatch):
    """A strategy that always HOLDs should produce zero trades."""
    monkeypatch.setattr(backtester, "fetch_ohlcv", lambda *a, **kw: synthetic_ohlcv)

    def hold_strategy(df, i, state):
        return {"action": "HOLD", "confidence": 0.0}

    result = backtester.run(
        "BTC/USDT", "2024-01-01", "2024-01-20",
        hold_strategy, timeframe="1h",
    )
    assert result.total_trades == 0


def test_always_buy_strategy_produces_trades(backtester, synthetic_ohlcv, monkeypatch):
    monkeypatch.setattr(backtester, "fetch_ohlcv", lambda *a, **kw: synthetic_ohlcv)

    def buy_strategy(df, i, state):
        return {
            "action": "BUY",
            "confidence": 0.8,
            "sl_distance_atr": 1.0,
            "tp_distance_atr": 1.5,
        }

    result = backtester.run(
        "BTC/USDT", "2024-01-01", "2024-01-20",
        buy_strategy, timeframe="1h",
        warmup_bars=50,
    )
    assert result.total_trades > 0
    assert len(result.equity_curve) > 0


def test_pnl_accounts_for_spread_and_fees(backtester, synthetic_ohlcv, monkeypatch):
    monkeypatch.setattr(backtester, "fetch_ohlcv", lambda *a, **kw: synthetic_ohlcv)

    def buy_strategy(df, i, state):
        return {"action": "BUY", "confidence": 0.8,
                "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}

    # Run with no fees
    no_fee = backtester.run(
        "BTC/USDT", "2024-01-01", "2024-01-20", buy_strategy,
        timeframe="1h", warmup_bars=50,
        spread_pct=0.0, commission_pct=0.0,
    )
    # Run with realistic fees
    with_fee = backtester.run(
        "BTC/USDT", "2024-01-01", "2024-01-20", buy_strategy,
        timeframe="1h", warmup_bars=50,
        spread_pct=0.05, commission_pct=0.1,
    )

    # Fees should reduce PnL
    if no_fee.total_trades > 0 and with_fee.total_trades > 0:
        assert with_fee.total_fees > 0
        assert with_fee.total_pnl < no_fee.total_pnl


def test_result_has_all_metrics(backtester, synthetic_ohlcv, monkeypatch):
    monkeypatch.setattr(backtester, "fetch_ohlcv", lambda *a, **kw: synthetic_ohlcv)

    def simple_strategy(df, i, state):
        if i % 30 == 0:
            return {"action": "BUY", "confidence": 0.7,
                    "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}
        return {"action": "HOLD", "confidence": 0.0}

    result = backtester.run(
        "BTC/USDT", "2024-01-01", "2024-01-20",
        simple_strategy, timeframe="1h", warmup_bars=50,
    )
    d = result.to_dict()
    for key in ["total_trades", "win_rate", "sharpe", "sortino", "max_drawdown"]:
        assert key in d


def test_empty_dataframe_returns_empty_result(backtester, monkeypatch):
    monkeypatch.setattr(
        backtester, "fetch_ohlcv", lambda *a, **kw: pd.DataFrame()
    )

    def strat(df, i, state):
        return {"action": "BUY", "confidence": 0.8}

    result = backtester.run("BTC/USDT", "2024-01-01", "2024-01-02", strat)
    assert result.total_trades == 0


def test_max_drawdown_positive():
    """Max drawdown calculation is non-negative."""
    equity = [
        {"timestamp": "t1", "balance": 10000},
        {"timestamp": "t2", "balance": 11000},
        {"timestamp": "t3", "balance": 10500},
        {"timestamp": "t4", "balance": 9500},
        {"timestamp": "t5", "balance": 10200},
    ]
    dd = HistoricalBacktester._max_drawdown(equity)
    # Peak was 11000, low was 9500 → DD = (11000-9500)/11000 = 0.1364
    assert 0.13 < dd < 0.14


def test_sharpe_handles_constant_returns():
    returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    # All-positive constant returns → std = 0 → Sharpe = 0 (by our guard)
    sharpe = HistoricalBacktester._sharpe(returns)
    assert sharpe == 0.0


def test_sharpe_reasonable_for_positive_returns():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, 100)  # slight positive drift
    sharpe = HistoricalBacktester._sharpe(returns)
    # With slight positive drift, Sharpe should be positive but small
    assert sharpe > -5 and sharpe < 20


def test_walk_forward_produces_multiple_results(backtester, synthetic_ohlcv, monkeypatch):
    monkeypatch.setattr(backtester, "fetch_ohlcv", lambda *a, **kw: synthetic_ohlcv)

    def always_hold(df, i, state):
        return {"action": "HOLD", "confidence": 0.0}

    # 500 hourly bars ≈ 20 days; use short windows
    results = backtester.run_walk_forward(
        "BTC/USDT", "2024-01-01", "2024-01-21",
        always_hold, train_days=5, test_days=5,
        timeframe="1h", warmup_bars=10,
    )
    assert len(results) >= 1
    assert all(isinstance(r, BacktestResult) for r in results)


def test_estimate_bars_1m():
    # 1 day = 1440 1m bars
    assert HistoricalBacktester._estimate_bars(
        "2024-01-01", "2024-01-02", "1m"
    ) == 1440


def test_estimate_bars_1h():
    assert HistoricalBacktester._estimate_bars(
        "2024-01-01", "2024-01-02", "1h"
    ) == 24


def test_coverage_check_rejects_empty():
    bt = HistoricalBacktester(cache_ohlcv=False)
    assert bt._coverage_ok(pd.DataFrame(), "2024-01-01", "2024-01-02", "1h") is False
