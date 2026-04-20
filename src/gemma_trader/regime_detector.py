"""
HMM Regime Detector
====================
Hidden Markov Model for market regime detection.

Replaces the crude 3-bucket vol_trend with 4 latent states fit to
(returns, volatility, volume_zscore). States emerge from data rather
than being hand-specified thresholds.

Typical emergent states (interpretation depends on training data):
- 0: Trending up (positive drift, low vol)
- 1: Trending down (negative drift, low vol)
- 2: High volatility chop (high vol, no drift)
- 3: Low volatility consolidation (low vol, no drift)

State labels are assigned post-hoc based on mean return + volatility
of each cluster.

Fallback: if hmmlearn is not installed, degrades to a rule-based
3-state classifier (high_vol, trending, ranging).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    HMM-based regime detector. Fits on historical OHLCV, predicts on live.
    """

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.model = None
        self.state_labels: dict[int, str] = {}
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    # ── Feature preparation ──

    @staticmethod
    def _prepare_features(df: pd.DataFrame) -> np.ndarray:
        """
        Extract [return, volatility, volume_zscore] per bar.
        Returns array of shape (n_bars, 3) after dropping warmup NaNs.
        """
        returns = df["close"].pct_change()
        vol_20 = returns.rolling(20).std()
        volume_mean = df["volume"].rolling(20).mean()
        volume_std = df["volume"].rolling(20).std()
        volume_zscore = (df["volume"] - volume_mean) / volume_std

        features = np.column_stack([
            returns.values,
            vol_20.values,
            volume_zscore.values,
        ])

        # Drop rows with any NaN
        valid_mask = ~np.isnan(features).any(axis=1)
        return features[valid_mask]

    # ── Train ──

    def fit(self, df: pd.DataFrame) -> bool:
        """
        Fit HMM on historical data. Requires hmmlearn.
        Returns True if fit succeeded, False if fell back to rule-based.
        """
        features = self._prepare_features(df)
        if len(features) < 500:
            logger.warning(f"Too few samples for HMM fit: {len(features)} < 500")
            return False

        # Standardize
        self.feature_means = features.mean(axis=0)
        self.feature_stds = features.std(axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0
        features_norm = (features - self.feature_means) / self.feature_stds

        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed; using rule-based fallback")
            self.model = None
            return False

        try:
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=50,
                random_state=42,
            )
            self.model.fit(features_norm)

            # Label states by (mean_return, mean_volatility)
            states = self.model.predict(features_norm)
            self._label_states(features, states)

            logger.info(f"HMM fit on {len(features)} bars, states: {self.state_labels}")
            return True
        except Exception as e:
            logger.error(f"HMM fit failed: {e}")
            self.model = None
            return False

    def _label_states(self, features: np.ndarray, states: np.ndarray) -> None:
        """
        Label each state cluster with a human-readable tag based on
        mean return and volatility within cluster.
        """
        labels = {}
        for state_id in range(self.n_states):
            mask = states == state_id
            if not mask.any():
                labels[state_id] = "unknown"
                continue
            mean_ret = features[mask, 0].mean()
            mean_vol = features[mask, 1].mean()

            vol_threshold = np.median(features[:, 1])

            if mean_vol > vol_threshold * 1.3:
                label = "high_vol_chop"
            elif mean_ret > 0.0001:
                label = "trending_up"
            elif mean_ret < -0.0001:
                label = "trending_down"
            else:
                label = "low_vol_range"
            labels[state_id] = label
        self.state_labels = labels

    # ── Predict ──

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return state index per bar (after warmup)."""
        features = self._prepare_features(df)
        if self.model is None or len(features) == 0:
            return self._rule_based_predict(df)

        features_norm = (features - self.feature_means) / self.feature_stds
        return self.model.predict(features_norm)

    def current_state(self, df: pd.DataFrame) -> dict:
        """
        Get the current (latest bar) regime state.
        Returns {"state_id", "label", "probabilities"}.
        """
        if self.model is None:
            label = self._rule_based_current(df)
            return {
                "state_id": -1,
                "label": label,
                "probabilities": {},
            }

        features = self._prepare_features(df)
        if len(features) == 0:
            return {"state_id": -1, "label": "unknown", "probabilities": {}}

        features_norm = (features - self.feature_means) / self.feature_stds
        probs = self.model.predict_proba(features_norm[-1:].reshape(1, -1))
        state_id = int(np.argmax(probs))
        label = self.state_labels.get(state_id, "unknown")

        prob_dict = {
            self.state_labels.get(i, f"state_{i}"): float(probs[0, i])
            for i in range(self.n_states)
        }

        return {
            "state_id": state_id,
            "label": label,
            "probabilities": prob_dict,
        }

    # ── Rule-based fallback ──

    @staticmethod
    def _rule_based_predict(df: pd.DataFrame) -> np.ndarray:
        """Fallback state labels: 0=range, 1=trend, 2=high_vol."""
        returns = df["close"].pct_change().fillna(0)
        vol = returns.rolling(20).std().fillna(0)
        vol_median = vol.median() if len(vol) > 0 else 0.0

        states = np.zeros(len(df), dtype=int)
        if vol_median > 0:
            states[vol > vol_median * 1.5] = 2
            trend_mask = returns.rolling(10).mean().abs() > returns.rolling(50).std()
            states[trend_mask.fillna(False).values & (states != 2)] = 1
        return states

    def _rule_based_current(self, df: pd.DataFrame) -> str:
        if len(df) < 30:
            return "unknown"
        returns = df["close"].pct_change().fillna(0)
        recent_vol = returns.tail(20).std()
        historical_vol = returns.std()
        if historical_vol == 0:
            return "low_vol_range"
        if recent_vol > historical_vol * 1.5:
            return "high_vol_chop"
        recent_ret = returns.tail(10).mean()
        if recent_ret > historical_vol * 0.3:
            return "trending_up"
        if recent_ret < -historical_vol * 0.3:
            return "trending_down"
        return "low_vol_range"

    # ── Persistence ──

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "state_labels": self.state_labels,
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds,
                "n_states": self.n_states,
            }, f)
        logger.info(f"Saved HMM model to {path}")

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.state_labels = data["state_labels"]
            self.feature_means = data["feature_means"]
            self.feature_stds = data["feature_stds"]
            self.n_states = data["n_states"]
            logger.info(f"Loaded HMM model from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load HMM model: {e}")
            return False


# ── Singleton helper ──

_detector: Optional[RegimeDetector] = None


def get_regime_detector(
    model_path: Optional[Path] = None,
    n_states: int = 4,
) -> RegimeDetector:
    global _detector
    if _detector is None:
        _detector = RegimeDetector(n_states=n_states)
        if model_path:
            _detector.load(model_path)
    return _detector
