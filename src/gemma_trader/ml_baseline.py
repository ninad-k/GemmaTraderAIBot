"""
XGBoost ML Baseline
=====================
Classical ML baseline that runs ALONGSIDE Gemma 4 to validate edge.

Why: If a tiny XGBoost model matches or beats a 9.6GB LLM at predicting
trade outcomes, your edge is in the features, not the model. This is
critical information — it tells you where to invest engineering effort.

The baseline trains on `trade_outcomes` (labeled WIN/LOSS) with all
computed indicators as features. Outputs a win probability per trade
attempt, shown on the dashboard next to Gemma's confidence.

Agreement-based gate:
- Gemma BUY + XGBoost prob_win > 0.6 → high-confidence trade
- Gemma BUY + XGBoost prob_win < 0.4 → veto (log, don't trade)
- Disagreement logged for post-mortem analysis
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Features used as ML inputs (stable naming contract)
FEATURE_COLUMNS = [
    # Momentum
    "rsi", "macd_hist", "stoch_k", "stoch_d", "cci", "williams_r", "roc", "mfi",
    # Trend
    "adx", "di_plus", "di_minus", "ema_diff_9_20", "ema_diff_20_50",
    # Volatility
    "atr", "bb_width", "bb_pos",
    # Advanced
    "order_flow_imbalance", "spread_atr_ratio", "volume_profile_zscore",
    "ret_autocorr_5", "realized_volatility_1h", "hurst_exponent",
    "dist_from_60high_pct", "dist_from_60low_pct",
    # Time
    "hour_of_day", "day_of_week", "session",
    # Position context
    "confidence",
]


class MLBaseline:
    """XGBoost classifier on trade outcomes."""

    def __init__(self):
        self.model = None
        self.feature_cols: list[str] = []
        self.feature_importances_: Optional[dict] = None
        self.metrics: Optional[dict] = None

    # ── Dataset preparation ──

    def prepare_dataset(
        self,
        outcomes: list[dict],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Flatten outcomes into X (features) and y (WIN=1, LOSS=0).
        Features are pulled from indicators_snapshot + top-level fields.
        """
        rows = []
        labels = []

        for o in outcomes:
            snap = o.get("indicators_snapshot", {}) or {}
            if isinstance(snap, str):
                try:
                    snap = json.loads(snap)
                except Exception:
                    snap = {}

            row = {}
            for col in FEATURE_COLUMNS:
                if col in snap:
                    row[col] = snap[col]
                elif col in o:
                    row[col] = o[col]
                else:
                    row[col] = np.nan

            # Encode action as feature (BUY=1, SELL=-1)
            action = o.get("action", "").upper()
            row["action_encoded"] = 1 if action == "BUY" else (-1 if action == "SELL" else 0)

            rows.append(row)
            result = o.get("result", "").upper()
            labels.append(1 if result == "WIN" else 0)

        X = pd.DataFrame(rows)
        y = pd.Series(labels, name="result")
        return X, y

    # ── Training ──

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        walk_forward: bool = True,
        n_splits: int = 5,
    ) -> dict:
        """
        Train with walk-forward CV. Returns metrics dict.
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                roc_auc_score, f1_score,
            )
        except ImportError as e:
            logger.error(f"ML deps missing: {e}. Run: pip install xgboost scikit-learn")
            return {"error": "dependencies_missing"}

        # Drop rows with too many NaN
        X = X.dropna(axis=0, thresh=len(X.columns) // 2)
        y = y.loc[X.index]

        if len(X) < 50:
            logger.warning(f"Too few samples for training: {len(X)} < 50")
            return {"error": "insufficient_data", "n_samples": len(X)}

        # Fill remaining NaN with column medians
        X = X.fillna(X.median(numeric_only=True))

        self.feature_cols = list(X.columns)

        if walk_forward and len(X) >= n_splits * 20:
            splits = TimeSeriesSplit(n_splits=n_splits)
            train_scores, test_scores = [], []

            for train_idx, test_idx in splits.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                )
                model.fit(X_tr, y_tr)

                y_pred_tr = model.predict(X_tr)
                y_pred_te = model.predict(X_te)
                train_scores.append(accuracy_score(y_tr, y_pred_tr))
                test_scores.append(accuracy_score(y_te, y_pred_te))

            # Train final model on all data
            self.model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric="logloss",
            )
            self.model.fit(X, y)

            y_pred = self.model.predict(X)
            y_proba = self.model.predict_proba(X)[:, 1]

            self.metrics = {
                "n_samples": len(X),
                "cv_train_acc_mean": float(np.mean(train_scores)),
                "cv_test_acc_mean": float(np.mean(test_scores)),
                "cv_test_acc_std": float(np.std(test_scores)),
                "final_accuracy": float(accuracy_score(y, y_pred)),
                "final_precision": float(precision_score(y, y_pred, zero_division=0)),
                "final_recall": float(recall_score(y, y_pred, zero_division=0)),
                "final_f1": float(f1_score(y, y_pred, zero_division=0)),
                "final_roc_auc": (
                    float(roc_auc_score(y, y_proba))
                    if len(np.unique(y)) > 1 else 0.0
                ),
            }
        else:
            # Single train/test split
            split = int(len(X) * 0.8)
            X_tr, X_te = X.iloc[:split], X.iloc[split:]
            y_tr, y_te = y.iloc[:split], y.iloc[split:]

            self.model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=42, eval_metric="logloss",
            )
            self.model.fit(X_tr, y_tr)

            y_pred = self.model.predict(X_te)
            self.metrics = {
                "n_samples": len(X),
                "accuracy": float(accuracy_score(y_te, y_pred)),
            }

        # Feature importance
        self.feature_importances_ = dict(
            zip(self.feature_cols, self.model.feature_importances_)
        )
        return self.metrics

    # ── Inference ──

    def predict(self, market_data: dict) -> dict:
        """
        Predict win probability for current market state.
        Returns {prob_win, prob_loss, top_features}.
        """
        if self.model is None:
            return {"prob_win": 0.5, "prob_loss": 0.5, "top_features": {}}

        # Build feature vector
        row = {}
        for col in self.feature_cols:
            if col in market_data:
                row[col] = market_data[col]
            elif col == "action_encoded":
                action = str(market_data.get("action", "")).upper()
                row[col] = 1 if action == "BUY" else (-1 if action == "SELL" else 0)
            else:
                row[col] = np.nan

        X = pd.DataFrame([row])
        X = X.fillna(0)  # no medians at inference; use 0

        proba = self.model.predict_proba(X)[0]
        prob_win = float(proba[1]) if len(proba) > 1 else 0.5

        top_features = {}
        if self.feature_importances_:
            sorted_imp = sorted(
                self.feature_importances_.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            top_features = {k: float(v) for k, v in sorted_imp}

        return {
            "prob_win": prob_win,
            "prob_loss": 1.0 - prob_win,
            "top_features": top_features,
        }

    # ── Agreement gate ──

    @staticmethod
    def agreement_gate(
        gemma_decision: dict,
        ml_prediction: dict,
        min_ml_prob: float = 0.55,
    ) -> tuple[bool, str]:
        """
        Consensus check: Gemma says X, ML says Y — agree?
        Returns (trade_allowed, reason).
        """
        action = str(gemma_decision.get("action", "")).upper()
        if action not in ("BUY", "SELL"):
            return True, "gemma_hold_not_gated"

        prob_win = ml_prediction.get("prob_win", 0.5)

        if prob_win >= min_ml_prob:
            return True, f"ml_confirms (prob_win={prob_win:.2f})"
        else:
            return False, f"ml_vetoes (prob_win={prob_win:.2f} < {min_ml_prob})"

    # ── Persistence ──

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_cols": self.feature_cols,
                "feature_importances_": self.feature_importances_,
                "metrics": self.metrics,
            }, f)
        logger.info(f"Saved ML baseline to {path}")

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feature_cols = data["feature_cols"]
            self.feature_importances_ = data.get("feature_importances_")
            self.metrics = data.get("metrics")
            logger.info(f"Loaded ML baseline from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load ML baseline: {e}")
            return False


# ── CLI ──

if __name__ == "__main__":
    import argparse
    from gemma_trader.storage import get_db

    parser = argparse.ArgumentParser(description="Train ML baseline")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model-path", default="logs/ml_baseline.pkl")
    parser.add_argument("--min-samples", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.train:
        db = get_db()
        outcomes = db.query_outcomes(limit=5000)
        if len(outcomes) < args.min_samples:
            print(f"Need at least {args.min_samples} outcomes (have {len(outcomes)})")
            exit(1)

        baseline = MLBaseline()
        X, y = baseline.prepare_dataset(outcomes)
        metrics = baseline.train(X, y, walk_forward=True)
        baseline.save(Path(args.model_path))

        print(json.dumps(metrics, indent=2))
        print("\nTop 10 feature importances:")
        for k, v in sorted(
            (baseline.feature_importances_ or {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]:
            print(f"  {k:30s} {v:.4f}")
