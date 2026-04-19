"""
Extra market features used by the trader beyond pandas_ta indicators:
- Funding rate (Binance futures, public endpoint)
- Order-book imbalance (Binance depth, public endpoint)
- BTC dominance (CoinGecko global, public)
- Correlation guard: compute rolling correlation between an incoming symbol
  and the currently-open book so we block piling into correlated exposure.

All calls are best-effort with short timeouts; on failure they return
neutral values so the decision pipeline is unaffected.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import requests

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "rey-capital-bot/1.0"})

BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/premiumIndex"
BINANCE_DEPTH = "https://fapi.binance.com/fapi/v1/depth"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"


def fetch_funding_rate(symbol: str) -> Optional[float]:
    """Return last funding rate (signed decimal) or None."""
    sym = symbol.replace("_", "").upper()
    try:
        r = _SESSION.get(BINANCE_FUNDING, params={"symbol": sym}, timeout=4)
        r.raise_for_status()
        return float(r.json().get("lastFundingRate", 0))
    except Exception as e:
        logger.debug(f"funding rate fetch failed for {symbol}: {e}")
        return None


def fetch_order_book_imbalance(symbol: str, levels: int = 20) -> Optional[float]:
    """
    Returns (bid_vol - ask_vol) / (bid_vol + ask_vol) in [-1, 1].
    Positive = buy pressure, negative = sell pressure.
    """
    sym = symbol.replace("_", "").upper()
    try:
        r = _SESSION.get(BINANCE_DEPTH, params={"symbol": sym, "limit": levels}, timeout=4)
        r.raise_for_status()
        data = r.json()
        bid_vol = sum(float(q) for _, q in data.get("bids", []))
        ask_vol = sum(float(q) for _, q in data.get("asks", []))
        total = bid_vol + ask_vol
        if total <= 0:
            return None
        return round((bid_vol - ask_vol) / total, 4)
    except Exception as e:
        logger.debug(f"orderbook fetch failed for {symbol}: {e}")
        return None


def fetch_btc_dominance() -> Optional[float]:
    try:
        r = _SESSION.get(COINGECKO_GLOBAL, timeout=6)
        r.raise_for_status()
        return round(float(r.json()["data"]["market_cap_percentage"]["btc"]), 3)
    except Exception as e:
        logger.debug(f"btc dominance fetch failed: {e}")
        return None


def correlation_ok(
    candidate_returns: Iterable[float],
    open_book_returns: dict[str, Iterable[float]],
    max_corr: float = 0.85,
) -> tuple[bool, dict]:
    """
    Check Pearson correlation of the candidate's recent returns against every
    already-open symbol. Return (ok, details). `ok` is False if any pair
    exceeds max_corr (threshold means "too correlated to add more exposure").
    """
    cand = list(candidate_returns)
    if len(cand) < 10:
        return True, {"reason": "insufficient samples"}
    details = {}
    for sym, series in open_book_returns.items():
        other = list(series)[-len(cand):]
        if len(other) < 10:
            continue
        c = _pearson(cand[-len(other):], other)
        details[sym] = round(c, 3)
        if abs(c) >= max_corr:
            return False, details
    return True, details


def _pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[-n:], b[-n:]
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a)
    db = sum((y - mb) ** 2 for y in b)
    denom = (da * db) ** 0.5
    return 0.0 if denom == 0 else num / denom
