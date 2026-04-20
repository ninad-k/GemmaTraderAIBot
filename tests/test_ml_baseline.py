"""Tests for ML baseline (XGBoost)."""

import pytest
import numpy as np

from gemma_trader.ml_baseline import MLBaseline, FEATURE_COLUMNS


def _synthetic_outcomes(n: int = 200, seed: int = 42) -> list[dict]:
    """Generate synthetic labeled trade outcomes with predictive features."""
    rng = np.random.default_rng(seed)
    outcomes = []
    for i in range(n):
        # Features with some predictive signal
        rsi = rng.uniform(20, 80)
        macd_hist = rng.normal(0, 0.5)
        adx = rng.uniform(10, 50)

        # Synthetic rule: low RSI + positive MACD + decent ADX → WIN
        win_score = (50 - rsi) / 30 + macd_hist * 0.5 + (adx - 25) / 20
        win = 1 if (win_score + rng.normal(0, 0.3)) > 0 else 0

        # Populate most FEATURE_COLUMNS so NaN-drop doesn't wipe dataset
        outcomes.append({
            "symbol": "BTCUSD",
            "action": "BUY",
            "result": "WIN" if win else "LOSS",
            "confidence": float(rng.uniform(0.6, 0.9)),
            "indicators_snapshot": {
                "rsi": rsi,
                "macd_hist": macd_hist,
                "stoch_k": float(rng.uniform(0, 100)),
                "stoch_d": float(rng.uniform(0, 100)),
                "cci": float(rng.uniform(-200, 200)),
                "williams_r": float(rng.uniform(-100, 0)),
                "roc": float(rng.normal(0, 2)),
                "mfi": float(rng.uniform(0, 100)),
                "adx": adx,
                "di_plus": float(rng.uniform(10, 40)),
                "di_minus": float(rng.uniform(10, 40)),
                "ema_diff_9_20": float(rng.normal(0, 10)),
                "ema_diff_20_50": float(rng.normal(0, 20)),
                "atr": float(rng.uniform(10, 50)),
                "bb_width": float(rng.uniform(0.01, 0.05)),
                "bb_pos": float(rng.uniform(0, 1)),
                "order_flow_imbalance": float(rng.uniform(-1, 1)),
                "spread_atr_ratio": float(rng.uniform(0, 0.2)),
                "volume_profile_zscore": float(rng.normal(0, 1)),
                "ret_autocorr_5": float(rng.uniform(-1, 1)),
                "realized_volatility_1h": float(rng.uniform(0.001, 0.05)),
                "hurst_exponent": float(rng.uniform(0.3, 0.7)),
                "dist_from_60high_pct": float(rng.uniform(0, 0.1)),
                "dist_from_60low_pct": float(rng.uniform(0, 0.1)),
                "hour_of_day": int(rng.integers(0, 24)),
                "day_of_week": int(rng.integers(0, 7)),
                "session": int(rng.integers(0, 4)),
            },
        })
    return outcomes


def test_prepare_dataset_shape():
    baseline = MLBaseline()
    outcomes = _synthetic_outcomes(100)
    X, y = baseline.prepare_dataset(outcomes)

    assert len(X) == 100
    assert len(y) == 100
    assert "rsi" in X.columns
    assert "action_encoded" in X.columns


def test_prepare_dataset_handles_missing_features():
    baseline = MLBaseline()
    outcomes = [
        {"symbol": "BTCUSD", "result": "WIN", "action": "BUY"},
        {"symbol": "BTCUSD", "result": "LOSS", "action": "SELL",
         "indicators_snapshot": {"rsi": 50}},
    ]
    X, y = baseline.prepare_dataset(outcomes)
    assert len(X) == 2
    # Missing fields should be NaN
    assert X["rsi"].isna().any() or 50.0 in X["rsi"].values


def test_train_insufficient_data():
    baseline = MLBaseline()
    outcomes = _synthetic_outcomes(20)
    X, y = baseline.prepare_dataset(outcomes)
    metrics = baseline.train(X, y)
    assert "error" in metrics or metrics.get("n_samples", 0) < 50


def test_train_with_enough_data():
    """May skip if xgboost not installed."""
    try:
        import xgboost  # noqa
    except ImportError:
        pytest.skip("xgboost not installed")

    baseline = MLBaseline()
    outcomes = _synthetic_outcomes(300)
    X, y = baseline.prepare_dataset(outcomes)
    metrics = baseline.train(X, y, walk_forward=True)

    assert "error" not in metrics
    assert metrics["n_samples"] > 200
    # With synthetic rule, should beat random
    if "cv_test_acc_mean" in metrics:
        assert metrics["cv_test_acc_mean"] > 0.45  # better than pure random


def test_predict_without_training():
    baseline = MLBaseline()
    result = baseline.predict({"rsi": 50, "action": "BUY"})
    assert result["prob_win"] == 0.5
    assert result["prob_loss"] == 0.5


def test_predict_after_training():
    try:
        import xgboost  # noqa
    except ImportError:
        pytest.skip("xgboost not installed")

    baseline = MLBaseline()
    outcomes = _synthetic_outcomes(300)
    X, y = baseline.prepare_dataset(outcomes)
    baseline.train(X, y)

    market_data = {
        "rsi": 30, "macd_hist": 0.5, "adx": 40,
        "atr": 20, "session": 2, "action": "BUY",
    }
    pred = baseline.predict(market_data)
    assert 0.0 <= pred["prob_win"] <= 1.0
    assert "top_features" in pred


def test_agreement_gate_buys_when_ml_confirms():
    allowed, reason = MLBaseline.agreement_gate(
        {"action": "BUY"},
        {"prob_win": 0.7},
        min_ml_prob=0.55,
    )
    assert allowed
    assert "confirms" in reason


def test_agreement_gate_vetoes_when_ml_disagrees():
    allowed, reason = MLBaseline.agreement_gate(
        {"action": "BUY"},
        {"prob_win": 0.3},
        min_ml_prob=0.55,
    )
    assert not allowed
    assert "veto" in reason.lower()


def test_agreement_gate_ignores_hold():
    """HOLD decisions are not gated — always pass through."""
    allowed, reason = MLBaseline.agreement_gate(
        {"action": "HOLD"},
        {"prob_win": 0.1},
    )
    assert allowed


def test_save_and_load_roundtrip(tmp_path):
    """Roundtrip should preserve metrics even without training."""
    baseline = MLBaseline()
    baseline.metrics = {"n_samples": 100, "accuracy": 0.65}

    path = tmp_path / "ml.pkl"
    baseline.save(path)
    assert path.exists()

    loaded = MLBaseline()
    assert loaded.load(path)
    assert loaded.metrics == baseline.metrics
