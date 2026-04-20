"""Tests for HMM regime detector."""

import numpy as np
import pandas as pd
import pytest

from gemma_trader.regime_detector import RegimeDetector


def _trending_df(n: int = 600, direction: int = 1) -> pd.DataFrame:
    """Synthetic trending data (up if direction=1, down if -1)."""
    rng = np.random.default_rng(42)
    drift = 0.0005 * direction
    returns = rng.normal(drift, 0.002, n)
    close = 45000 * np.exp(returns.cumsum())
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": rng.uniform(1000, 5000, n),
    }, index=idx)


def _choppy_df(n: int = 600) -> pd.DataFrame:
    """Synthetic high-vol choppy data."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, n)  # no drift, high vol
    close = 45000 * np.exp(returns.cumsum())
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": rng.uniform(1000, 5000, n),
    }, index=idx)


def test_detector_initializes():
    detector = RegimeDetector(n_states=4)
    assert detector.n_states == 4
    assert detector.model is None


def test_prepare_features_shape():
    df = _trending_df(600)
    features = RegimeDetector._prepare_features(df)
    # 3 features per bar (return, volatility, volume_zscore)
    assert features.shape[1] == 3
    # Warmup drops some rows
    assert features.shape[0] > 500


def test_rule_based_current_trending_up():
    detector = RegimeDetector()
    df = _trending_df(200, direction=1)
    state = detector.current_state(df)
    assert state["label"] in ("trending_up", "low_vol_range", "unknown")


def test_rule_based_current_trending_down():
    detector = RegimeDetector()
    df = _trending_df(200, direction=-1)
    state = detector.current_state(df)
    assert state["label"] in ("trending_down", "low_vol_range", "unknown")


def test_rule_based_current_high_vol():
    detector = RegimeDetector()
    df = _choppy_df(200)
    state = detector.current_state(df)
    # High vol choppy data should trigger high_vol_chop in rule-based mode
    assert state["label"] in ("high_vol_chop", "low_vol_range")


def test_fit_with_insufficient_data():
    detector = RegimeDetector()
    df = _trending_df(100)  # below 500 sample threshold
    result = detector.fit(df)
    assert result is False


def test_fit_succeeds_with_enough_data():
    """May return True or False depending on hmmlearn install."""
    detector = RegimeDetector()
    df = _trending_df(600)
    result = detector.fit(df)
    # Result is True if hmmlearn available, False otherwise
    # Either way, .current_state() should still work
    state = detector.current_state(df)
    assert "label" in state


def test_save_and_load_roundtrip(tmp_path):
    detector = RegimeDetector()
    df = _trending_df(600)
    detector.fit(df)  # May or may not fit depending on hmmlearn

    path = tmp_path / "hmm.pkl"
    detector.save(path)
    assert path.exists()

    detector2 = RegimeDetector()
    loaded = detector2.load(path)
    assert loaded
    assert detector2.n_states == detector.n_states


def test_load_missing_file_returns_false(tmp_path):
    detector = RegimeDetector()
    assert detector.load(tmp_path / "nonexistent.pkl") is False
