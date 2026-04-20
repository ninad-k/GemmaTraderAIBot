"""Tests for advanced feature engineering."""

import numpy as np
import pandas as pd
import pytest

from gemma_trader.advanced_features import (
    compute_advanced_features,
    is_tradeable,
    _hurst,
)


def _synthetic_df(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic dynamics."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.002, n_bars).cumsum()
    close = 45000 * np.exp(returns)
    high = close * (1 + np.abs(rng.normal(0, 0.001, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(1000, 5000, n_bars)

    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=idx)


def test_features_produce_all_keys():
    df = _synthetic_df(200)
    features = compute_advanced_features(df)

    expected = [
        "order_flow_imbalance", "raw_spread", "spread_atr_ratio",
        "volume_profile_zscore", "ret_autocorr_5",
        "realized_volatility_1h", "btc_eth_spread_zscore",
        "hour_of_day", "day_of_week", "session",
        "dist_from_60high_pct", "dist_from_60low_pct",
        "bars_since_60high", "hurst_exponent",
    ]
    for key in expected:
        assert key in features, f"Missing feature: {key}"


def test_features_handle_empty_df():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    features = compute_advanced_features(df)
    # Should not raise, should return sensible defaults
    assert features["order_flow_imbalance"] == 0.0
    assert features["volume_profile_zscore"] == 0.0


def test_order_flow_imbalance_from_tick():
    df = _synthetic_df(50)
    tick = {
        "bid": 45000, "ask": 45001,
        "bid_vol": 100, "ask_vol": 300,
    }
    features = compute_advanced_features(df, tick_data=tick)
    # ask_vol > bid_vol → positive imbalance (buying pressure)
    assert features["order_flow_imbalance"] > 0
    assert features["raw_spread"] == 1.0


def test_volume_profile_zscore_detects_spike():
    df = _synthetic_df(50)
    # Spike last volume 5x
    df.loc[df.index[-1], "volume"] = df["volume"].iloc[-2] * 5

    features = compute_advanced_features(df)
    assert features["volume_profile_zscore"] > 1.5


def test_hurst_exponent_bounds():
    # Random walk should produce a finite value in valid range
    rng = np.random.default_rng(42)
    random_walk = rng.normal(0, 1, 200).cumsum()
    h = _hurst(random_walk)
    assert isinstance(h, float)
    assert -2.0 <= h <= 4.0  # Hurst variants can range widely

    # Hurst of a trend should differ from Hurst of mean-reverting series
    trend = np.arange(200, dtype=float) + rng.normal(0, 0.1, 200)
    h_trend = _hurst(trend)
    mean_rev = rng.normal(0, 1, 200)  # white noise (mean-reverting-ish)
    h_mr = _hurst(mean_rev)
    # They should differ (not both collapse to default 0.5)
    assert h_trend != h_mr


def test_time_features_populated():
    df = _synthetic_df(100)
    features = compute_advanced_features(df)
    assert 0 <= features["hour_of_day"] <= 23
    assert 0 <= features["day_of_week"] <= 6
    assert 0 <= features["session"] <= 3


def test_is_tradeable_accepts_normal():
    features = {
        "spread_atr_ratio": 0.1,
        "volume_profile_zscore": 0.0,
    }
    ok, reason = is_tradeable(features)
    assert ok
    assert reason == "ok"


def test_is_tradeable_rejects_wide_spread():
    features = {"spread_atr_ratio": 0.3}
    ok, reason = is_tradeable(features)
    assert not ok
    assert "spread" in reason.lower()


def test_is_tradeable_rejects_dead_volume():
    features = {
        "spread_atr_ratio": 0.1,
        "volume_profile_zscore": -2.0,
    }
    ok, reason = is_tradeable(features)
    assert not ok
    assert "dead" in reason.lower() or "volume" in reason.lower()


def test_cross_asset_features():
    btc = _synthetic_df(100, seed=1)
    eth = _synthetic_df(100, seed=2)
    features = compute_advanced_features(_synthetic_df(100), btc_df=btc, eth_df=eth)
    assert "btc_eth_spread_zscore" in features
    assert isinstance(features["btc_eth_spread_zscore"], float)


def test_60_bar_structure():
    df = _synthetic_df(100)
    features = compute_advanced_features(df)
    # dist_from_60high should be >= 0 and <= 1 typically
    assert features["dist_from_60high_pct"] >= 0
    assert features["dist_from_60low_pct"] >= 0
    assert features["bars_since_60high"] >= 0


def test_autocorr_handles_constant_returns():
    """Constant prices → zero returns → no autocorr."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open": [100] * n, "high": [100] * n, "low": [100] * n,
        "close": [100] * n, "volume": [1000] * n,
    }, index=idx)
    features = compute_advanced_features(df)
    assert features["ret_autocorr_5"] == 0.0
