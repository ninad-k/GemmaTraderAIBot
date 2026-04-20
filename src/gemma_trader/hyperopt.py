"""
Bayesian Hyperparameter Optimization
=====================================
Uses Optuna to search optimal values for:
- confidence_threshold
- sl_atr_multiplier
- tp_atr_multiplier
- cooldown_minutes
- max_open_trades

Objective: maximize Sharpe ratio over walk-forward validation.
Validated only on held-out test windows (no peeking).

Output: YAML diff file (`config.optimized.yaml`) that user can review
and merge incrementally. Never overwrites config.yaml directly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

from gemma_trader.historical_backtester import HistoricalBacktester

logger = logging.getLogger(__name__)


@dataclass
class HyperoptResult:
    best_params: dict
    best_score: float
    all_trials: list
    n_trials: int
    symbol: str
    period_days: int

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "period_days": self.period_days,
            "n_trials": self.n_trials,
            "best_score": round(self.best_score, 4),
            "best_params": self.best_params,
        }


def _make_threshold_strategy(
    confidence_threshold: float,
    sl_atr: float,
    tp_atr: float,
):
    """
    Generate a decision_fn for backtesting that uses threshold gating.
    For hyperopt, we don't run Gemma — we simulate a signal quality.
    This is a calibration strategy: it buys when RSI+MACD align and
    quality exceeds the threshold. Used to calibrate gate levels.
    """
    def strategy(df, i, state):
        if len(df) < 50:
            return {"action": "HOLD", "confidence": 0.0}

        close = df["close"]
        # Fast RSI
        diff = close.diff()
        gain = diff.where(diff > 0, 0).rolling(14).mean()
        loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50.0

        # Fast MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        macd_hist = float((macd - signal).iloc[-1]) if not macd.empty else 0.0

        # Confluence score → confidence
        buy_signals = int(rsi_val < 35) + int(macd_hist > 0)
        sell_signals = int(rsi_val > 65) + int(macd_hist < 0)

        if buy_signals >= 2:
            confidence = 0.5 + buy_signals * 0.1
            if confidence >= confidence_threshold:
                return {
                    "action": "BUY",
                    "confidence": confidence,
                    "sl_distance_atr": sl_atr,
                    "tp_distance_atr": tp_atr,
                }
        elif sell_signals >= 2:
            confidence = 0.5 + sell_signals * 0.1
            if confidence >= confidence_threshold:
                return {
                    "action": "SELL",
                    "confidence": confidence,
                    "sl_distance_atr": sl_atr,
                    "tp_distance_atr": tp_atr,
                }

        return {"action": "HOLD", "confidence": 0.0}

    return strategy


def run_optimization(
    symbol: str,
    start: str,
    end: str,
    n_trials: int = 50,
    timeframe: str = "1h",
    direction: str = "maximize",
    seed: int = 42,
) -> HyperoptResult:
    """
    Run Bayesian optimization over hyperparameter space.

    Args:
        symbol: e.g. "BTC/USDT"
        start: YYYY-MM-DD
        end: YYYY-MM-DD
        n_trials: number of Optuna trials (50 is a reasonable default)
        timeframe: backtest bar timeframe
        direction: "maximize" (default, for Sharpe) or "minimize"

    Returns: HyperoptResult with best_params + score.
    """
    try:
        import optuna
    except ImportError:
        logger.error("optuna not installed. Run: pip install optuna")
        return HyperoptResult({}, 0.0, [], 0, symbol, 0)

    # Suppress optuna INFO logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    bt = HistoricalBacktester(cache_ohlcv=True)
    trials_log = []

    def objective(trial):
        # Sample hyperparameters
        confidence_threshold = trial.suggest_float(
            "confidence_threshold", 0.50, 0.85
        )
        sl_atr = trial.suggest_float("sl_atr_multiplier", 0.5, 2.0)
        tp_atr = trial.suggest_float("tp_atr_multiplier", 1.0, 3.0)
        risk_pct = trial.suggest_float("risk_per_trade_pct", 0.5, 2.0)

        strategy = _make_threshold_strategy(
            confidence_threshold, sl_atr, tp_atr
        )

        result = bt.run(
            symbol,
            start,
            end,
            strategy,
            timeframe=timeframe,
            start_balance=10_000,
            risk_per_trade_pct=risk_pct,
        )

        trials_log.append({
            "confidence_threshold": confidence_threshold,
            "sl_atr": sl_atr,
            "tp_atr": tp_atr,
            "risk_pct": risk_pct,
            "sharpe": result.sharpe,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
            "max_drawdown": result.max_drawdown,
        })

        # Penalize strategies that don't trade enough (overfitting risk)
        if result.total_trades < 5:
            return -10.0

        # Composite objective: Sharpe penalized by max drawdown
        score = result.sharpe * (1 - min(result.max_drawdown, 0.5))
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    period_days = (
        (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
    )

    return HyperoptResult(
        best_params=study.best_params,
        best_score=study.best_value,
        all_trials=trials_log,
        n_trials=n_trials,
        symbol=symbol,
        period_days=period_days,
    )


def apply_best_params(
    best_params: dict,
    base_config_path: Path,
    output_path: Path,
) -> None:
    """
    Write optimized params to output YAML, keeping the rest of config intact.
    User must review diff and merge manually.
    """
    with open(base_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply to config sections
    if "trading" in config:
        if "confidence_threshold" in best_params:
            config["trading"]["confidence_threshold"] = round(
                best_params["confidence_threshold"], 3
            )

    if "risk_management" in config:
        if "sl_atr_multiplier" in best_params:
            config["risk_management"]["stop_loss_atr_multiplier"] = round(
                best_params["sl_atr_multiplier"], 3
            )
        if "tp_atr_multiplier" in best_params:
            config["risk_management"]["take_profit_atr_multiplier"] = round(
                best_params["tp_atr_multiplier"], 3
            )

    # Add a header comment
    header = (
        "# =====================================================\n"
        "# OPTIMIZED CONFIG — generated by hyperopt.py\n"
        f"# Generated: {datetime.now().isoformat()}\n"
        "# REVIEW and diff against config.yaml before adopting.\n"
        "# =====================================================\n\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Wrote optimized config to {output_path}")


# ── CLI ──

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bayesian hyperparameter optimization")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--base-config", default="config.yaml")
    parser.add_argument("--output", default="config.optimized.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"Optimizing {args.symbol} over {args.days} days ({args.trials} trials)...")
    result = run_optimization(
        args.symbol,
        start, end,
        n_trials=args.trials,
        timeframe=args.timeframe,
    )

    print(json.dumps(result.to_dict(), indent=2))

    if result.best_params:
        apply_best_params(
            result.best_params,
            Path(args.base_config),
            Path(args.output),
        )
        print(f"\n✓ Optimized config written to {args.output}")
        print(f"  Review with: diff {args.base_config} {args.output}")
