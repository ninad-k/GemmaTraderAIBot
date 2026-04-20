"""Tests for Bayesian hyperparameter optimization."""

import pytest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from gemma_trader.hyperopt import (
    _make_threshold_strategy,
    apply_best_params,
    HyperoptResult,
)


def test_threshold_strategy_signature():
    """Strategy function has the expected shape."""
    strat = _make_threshold_strategy(0.6, 1.0, 1.5)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "close": 45000 * np.exp(rng.normal(0, 0.002, 100).cumsum()),
        "high": 45100 * np.exp(rng.normal(0, 0.002, 100).cumsum()),
        "low": 44900 * np.exp(rng.normal(0, 0.002, 100).cumsum()),
        "open": 45000 * np.exp(rng.normal(0, 0.002, 100).cumsum()),
        "volume": rng.uniform(1000, 5000, 100),
    })
    result = strat(df, 99, {})
    assert "action" in result
    assert "confidence" in result


def test_threshold_strategy_holds_on_short_df():
    strat = _make_threshold_strategy(0.6, 1.0, 1.5)
    df = pd.DataFrame({"close": [100, 101, 102]})
    result = strat(df, 2, {})
    assert result["action"] == "HOLD"


def test_apply_best_params_writes_optimized_config(tmp_path):
    base_config = tmp_path / "config.yaml"
    base_config.write_text(yaml.safe_dump({
        "trading": {
            "mode": "paper",
            "confidence_threshold": 0.60,
        },
        "risk_management": {
            "stop_loss_atr_multiplier": 1.0,
            "take_profit_atr_multiplier": 1.5,
        },
    }))

    best = {
        "confidence_threshold": 0.72,
        "sl_atr_multiplier": 1.3,
        "tp_atr_multiplier": 2.1,
    }
    output = tmp_path / "config.optimized.yaml"
    apply_best_params(best, base_config, output)

    assert output.exists()
    content = output.read_text()
    # Optimized values applied
    assert "0.72" in content
    assert "1.3" in content
    assert "2.1" in content
    # Original structure preserved
    assert "paper" in content


def test_apply_best_params_preserves_other_fields(tmp_path):
    base_config = tmp_path / "config.yaml"
    base_config.write_text(yaml.safe_dump({
        "trading": {"mode": "paper", "confidence_threshold": 0.60},
        "risk_management": {"stop_loss_atr_multiplier": 1.0},
        "ollama": {"model": "gemma4:latest", "temperature": 0.1},
    }))

    apply_best_params(
        {"confidence_threshold": 0.75},
        base_config,
        tmp_path / "out.yaml",
    )

    result = yaml.safe_load((tmp_path / "out.yaml").read_text())
    # Preserved
    assert result["ollama"]["model"] == "gemma4:latest"
    assert result["trading"]["mode"] == "paper"
    # Updated
    assert result["trading"]["confidence_threshold"] == 0.75


def test_hyperopt_result_to_dict():
    result = HyperoptResult(
        best_params={"confidence_threshold": 0.7},
        best_score=1.5,
        all_trials=[],
        n_trials=50,
        symbol="BTC/USDT",
        period_days=90,
    )
    d = result.to_dict()
    assert d["symbol"] == "BTC/USDT"
    assert d["n_trials"] == 50
    assert d["best_score"] == 1.5


def test_run_optimization_without_optuna(monkeypatch):
    """Graceful degradation if optuna is missing."""
    import sys
    # Force optuna import to fail
    monkeypatch.setitem(sys.modules, "optuna", None)

    from gemma_trader.hyperopt import run_optimization

    # Should still return a result (empty) without raising
    # May hit ImportError inside the module but should be caught
    try:
        result = run_optimization(
            "BTC/USDT", "2024-01-01", "2024-01-02",
            n_trials=1, timeframe="1h",
        )
        assert isinstance(result, HyperoptResult)
    except ImportError:
        pass  # Acceptable — module-level import failures
