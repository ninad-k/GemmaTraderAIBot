"""
Advanced Feature Engineering
==============================
Microstructure and regime features that go beyond standard TA indicators.

The 30+ technical indicators in local_trader.py are largely redundant
(multiple EMAs, multiple momentum oscillators). Real alpha is in:

- Order flow imbalance (who's pushing price)
- Spread dynamics (liquidity regime)
- Realized volatility (true vol, not ATR proxy)
- Autocorrelation (trend vs. mean-reversion)
- Cross-asset lead/lag (BTC → alts)
- Session/time effects
- Hurst exponent (persistence)

These features feed into both the LLM context AND the ML baseline.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_advanced_features(
    df: pd.DataFrame,
    tick_data: Optional[dict] = None,
    btc_df: Optional[pd.DataFrame] = None,
    eth_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute advanced microstructure + regime features.

    Args:
        df: OHLCV DataFrame (required cols: open, high, low, close, volume)
        tick_data: Optional live tick snapshot {bid, ask, bid_vol, ask_vol}
        btc_df: Optional BTC OHLCV for cross-asset features (same timeframe)
        eth_df: Optional ETH OHLCV for cross-asset features

    Returns:
        Dict of advanced features ready to merge into market_data.
    """
    features = {}

    # ── Order flow imbalance (from tick, not candle) ──
    if tick_data:
        bid_vol = tick_data.get("bid_vol", 0)
        ask_vol = tick_data.get("ask_vol", 0)
        total = bid_vol + ask_vol
        features["order_flow_imbalance"] = (
            (ask_vol - bid_vol) / total if total > 0 else 0.0
        )
        bid = tick_data.get("bid", 0)
        ask = tick_data.get("ask", 0)
        features["raw_spread"] = ask - bid if ask and bid else 0.0
    else:
        features["order_flow_imbalance"] = 0.0
        features["raw_spread"] = 0.0

    # ── Spread / ATR ratio (trade-quality filter) ──
    atr = _atr(df, 14)
    features["spread_atr_ratio"] = (
        features["raw_spread"] / atr if atr > 0 else 0.0
    )

    # ── Volume profile z-score ──
    if len(df) >= 20 and "volume" in df.columns:
        vol_series = df["volume"].tail(20)
        vol_mean = vol_series.mean()
        vol_std = vol_series.std()
        current_vol = df["volume"].iloc[-1]
        features["volume_profile_zscore"] = (
            (current_vol - vol_mean) / vol_std if vol_std > 0 else 0.0
        )
    else:
        features["volume_profile_zscore"] = 0.0

    # ── Return autocorrelation (momentum vs. mean-reversion signal) ──
    returns = df["close"].pct_change().dropna()
    if len(returns) >= 20:
        recent = returns.tail(10).values
        prior = returns.iloc[-20:-10].values
        if len(recent) == len(prior) and np.std(recent) > 0 and np.std(prior) > 0:
            features["ret_autocorr_5"] = float(np.corrcoef(recent, prior)[0, 1])
        else:
            features["ret_autocorr_5"] = 0.0
    else:
        features["ret_autocorr_5"] = 0.0

    # ── Realized volatility (1h rolling, bar-frequency adjusted) ──
    if len(returns) >= 60:
        rv = returns.tail(60).std() * math.sqrt(60)
        features["realized_volatility_1h"] = float(rv)
    else:
        features["realized_volatility_1h"] = 0.0

    # ── Cross-asset features (BTC-ETH spread z-score) ──
    if btc_df is not None and eth_df is not None:
        features["btc_eth_spread_zscore"] = _btc_eth_spread(btc_df, eth_df)
    else:
        features["btc_eth_spread_zscore"] = 0.0

    # ── Time-based features ──
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        last_ts = df.index[-1]
        features["hour_of_day"] = int(last_ts.hour)
        features["day_of_week"] = int(last_ts.dayofweek)
        # Session encoding: 0=asia_overnight, 1=london, 2=ny_overlap, 3=ny
        hour = last_ts.hour
        if 0 <= hour < 7:
            session = 0
        elif 7 <= hour < 12:
            session = 1
        elif 12 <= hour < 16:
            session = 2
        else:
            session = 3
        features["session"] = session
    else:
        now = datetime.now()
        features["hour_of_day"] = now.hour
        features["day_of_week"] = now.weekday()
        features["session"] = 0

    # ── Structural features (pullback vs. breakout) ──
    if len(df) >= 60:
        high_60 = df["high"].tail(60).max()
        low_60 = df["low"].tail(60).min()
        current = df["close"].iloc[-1]
        features["dist_from_60high_pct"] = (
            (high_60 - current) / high_60 if high_60 > 0 else 0.0
        )
        features["dist_from_60low_pct"] = (
            (current - low_60) / low_60 if low_60 > 0 else 0.0
        )
        # Days since 60-bar high
        high_idx = df["high"].tail(60).idxmax()
        features["bars_since_60high"] = int(
            (df.index[-1] - high_idx).total_seconds() / 60
            if isinstance(df.index, pd.DatetimeIndex)
            else len(df) - df["high"].tail(60).values.argmax() - 1
        )
    else:
        features["dist_from_60high_pct"] = 0.0
        features["dist_from_60low_pct"] = 0.0
        features["bars_since_60high"] = 0

    # ── Hurst exponent (trending vs. mean-reverting) ──
    if len(df) >= 100:
        features["hurst_exponent"] = _hurst(df["close"].tail(100).values)
    else:
        features["hurst_exponent"] = 0.5  # neutral

    return features


# ── Internal helpers ──


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _btc_eth_spread(btc_df: pd.DataFrame, eth_df: pd.DataFrame, window: int = 60) -> float:
    """Z-score of BTC-ETH relative return spread."""
    if len(btc_df) < window or len(eth_df) < window:
        return 0.0
    btc_ret = btc_df["close"].pct_change().tail(window).dropna()
    eth_ret = eth_df["close"].pct_change().tail(window).dropna()
    n = min(len(btc_ret), len(eth_ret))
    if n < 10:
        return 0.0
    spread = btc_ret.values[-n:] - eth_ret.values[-n:]
    current = spread[-1]
    mean = np.mean(spread[:-1])
    std = np.std(spread[:-1])
    return float((current - mean) / std) if std > 0 else 0.0


def _hurst(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Hurst exponent via rescaled range analysis.
    H > 0.5 = trending, H < 0.5 = mean-reverting, H ≈ 0.5 = random walk.
    """
    try:
        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            diffs = np.subtract(series[lag:], series[:-lag])
            std = np.std(diffs)
            if std > 0:
                tau.append(std)
            else:
                tau.append(1e-10)
        if len(tau) < 2 or min(tau) <= 0:
            return 0.5
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        slope = np.polyfit(log_lags, log_tau, 1)[0]
        return float(slope * 2.0)  # Hurst = slope * 2 in R/S variant
    except Exception:
        return 0.5


# ── Quality filter (pre-trade gate) ──


def is_tradeable(features: dict) -> tuple[bool, str]:
    """
    Pre-trade quality gate based on advanced features.
    Returns (tradeable, reason) — reject trades in poor conditions.
    """
    # Spread too wide relative to target volatility
    if features.get("spread_atr_ratio", 0) > 0.25:
        return False, "spread_atr_ratio > 0.25 (TP structurally unreachable)"

    # Dead volume regime
    if features.get("volume_profile_zscore", 0) < -1.5:
        return False, "volume_profile_zscore < -1.5 (dead market)"

    # Asia overnight trap (hour 0-6 UTC, often low edge for retail)
    # Soft-gate: only warn, not block
    return True, "ok"
