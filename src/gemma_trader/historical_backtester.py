"""
Historical Backtester
======================
Proper backtesting on arbitrary historical date ranges.

Unlike the legacy backtester.py (which only replays logged outcomes),
this fetches real OHLCV data, walks bar-by-bar, and simulates any
strategy function with configurable slippage + commission modelling.

Data sources:
- ccxt (Binance, Coinbase, Kraken, etc. — crypto)
- Local CSV (user-provided historical dumps)
- SQLite cache (fetched data is cached 24h)

Usage:
    from gemma_trader.historical_backtester import HistoricalBacktester

    bt = HistoricalBacktester()
    result = bt.run(
        symbol="BTC/USDT",
        start="2024-01-01", end="2024-06-01",
        timeframe="1m",
        strategy_fn=my_strategy,
        spread_pct=0.05,
        commission_pct=0.01,
    )
    print(f"Sharpe: {result.sharpe:.2f}, MaxDD: {result.max_drawdown:.1%}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from gemma_trader.storage import get_db

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbol: str
    start: str
    end: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    final_balance: float = 0.0
    start_balance: float = 0.0
    total_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "period": f"{self.start} → {self.end}",
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "final_balance": round(self.final_balance, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_fees": round(self.total_fees, 2),
            "profit_factor": round(self.profit_factor, 3),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "max_drawdown": round(self.max_drawdown, 4),
        }


class HistoricalBacktester:
    """Replay arbitrary strategies on historical OHLCV data."""

    def __init__(self, cache_ohlcv: bool = True):
        self.cache_ohlcv = cache_ohlcv
        self.db = get_db() if cache_ohlcv else None

    # ── Data fetching ──

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV from cache first, then ccxt if not cached.
        Returns DataFrame indexed by timestamp with [open, high, low, close, volume].
        """
        # Try cache
        if self.db:
            cached = self.db.query_ohlcv(symbol, timeframe, start, end)
            if cached:
                df = pd.DataFrame(cached)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()
                if self._coverage_ok(df, start, end, timeframe):
                    logger.info(f"Cache hit: {len(df)} bars for {symbol}")
                    return df

        # Fetch from ccxt
        df = self._fetch_from_ccxt(symbol, start, end, timeframe)

        # Write to cache
        if self.db and len(df):
            candles = [
                {
                    "timestamp": ts.isoformat(),
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
                for ts, row in df.iterrows()
            ]
            inserted = self.db.cache_ohlcv(symbol, timeframe, candles)
            logger.info(f"Cached {inserted} bars for {symbol}")

        return df

    def _coverage_ok(self, df: pd.DataFrame, start: str, end: str, tf: str) -> bool:
        """Check if cached data has enough coverage."""
        if df.empty:
            return False
        expected_bars = self._estimate_bars(start, end, tf)
        # Allow 10% tolerance (gaps, weekends, etc.)
        return len(df) >= expected_bars * 0.85

    @staticmethod
    def _estimate_bars(start: str, end: str, tf: str) -> int:
        tf_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440,
        }.get(tf, 1)
        dt_start = pd.to_datetime(start)
        dt_end = pd.to_datetime(end)
        minutes = (dt_end - dt_start).total_seconds() / 60
        return int(minutes / tf_minutes)

    def _fetch_from_ccxt(
        self, symbol: str, start: str, end: str, timeframe: str
    ) -> pd.DataFrame:
        """Fetch OHLCV via ccxt. Symbol should be in ccxt format (BTC/USDT)."""
        try:
            import ccxt
        except ImportError:
            logger.error("ccxt not installed. Run: pip install ccxt")
            return pd.DataFrame()

        exchange = ccxt.binance({"enableRateLimit": True})
        since = int(pd.to_datetime(start).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end).timestamp() * 1000)

        all_candles = []
        while since < end_ts:
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not batch:
                    break
                all_candles.extend(batch)
                since = batch[-1][0] + 1
                if len(batch) < 1000:
                    break
            except Exception as e:
                logger.warning(f"ccxt fetch failed: {e}")
                break

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df

    # ── Core backtest loop ──

    def run(
        self,
        symbol: str,
        start: str,
        end: str,
        strategy_fn: Callable[[pd.DataFrame, int, dict], dict],
        *,
        timeframe: str = "1m",
        start_balance: float = 10_000,
        risk_per_trade_pct: float = 1.0,
        spread_pct: float = 0.05,
        commission_pct: float = 0.01,
        warmup_bars: int = 200,
    ) -> BacktestResult:
        """
        Run backtest for a symbol over a date range.

        strategy_fn signature: (df, current_idx, state) -> {
            "action": "BUY"|"SELL"|"HOLD",
            "confidence": float,
            "sl_distance_atr": float,
            "tp_distance_atr": float,
        }
        """
        df = self.fetch_ohlcv(symbol, start, end, timeframe)
        if df.empty:
            logger.warning(f"No data for {symbol}")
            return BacktestResult(symbol=symbol, start=start, end=end)

        balance = start_balance
        position: Optional[dict] = None
        result = BacktestResult(
            symbol=symbol,
            start=start,
            end=end,
            start_balance=start_balance,
        )
        equity_points = []
        pnl_series = []
        state: dict = {}

        for i in range(warmup_bars, len(df)):
            bar = df.iloc[i]
            close = bar["close"]
            high = bar["high"]
            low = bar["low"]

            # ── Check exit for open position ──
            if position:
                exit_price = None
                hit = None
                if position["side"] == "BUY":
                    if low <= position["sl"]:
                        exit_price = position["sl"]
                        hit = "sl"
                    elif high >= position["tp"]:
                        exit_price = position["tp"]
                        hit = "tp"
                else:  # SELL
                    if high >= position["sl"]:
                        exit_price = position["sl"]
                        hit = "sl"
                    elif low <= position["tp"]:
                        exit_price = position["tp"]
                        hit = "tp"

                if exit_price is not None:
                    pnl = self._compute_pnl(
                        position, exit_price, spread_pct, commission_pct
                    )
                    balance += pnl
                    pnl_series.append(pnl)
                    result.total_trades += 1
                    if pnl > 0:
                        result.wins += 1
                    else:
                        result.losses += 1
                    result.gross_pnl += pnl + position.get("fees", 0)
                    result.total_fees += position.get("fees", 0)
                    result.trades.append({
                        "entry_time": position["entry_time"].isoformat(),
                        "exit_time": df.index[i].isoformat(),
                        "side": position["side"],
                        "entry": position["entry"],
                        "exit": exit_price,
                        "qty": position["qty"],
                        "pnl": round(pnl, 4),
                        "hit": hit,
                    })
                    position = None

            # ── New entry signal ──
            if position is None:
                window = df.iloc[: i + 1]
                decision = strategy_fn(window, i, state)
                action = decision.get("action", "HOLD")

                if action in ("BUY", "SELL"):
                    atr = self._compute_atr(window, period=14)
                    if atr <= 0 or math.isnan(atr):
                        continue
                    sl_dist = decision.get("sl_distance_atr", 1.0) * atr
                    tp_dist = decision.get("tp_distance_atr", 1.5) * atr

                    risk_amount = balance * (risk_per_trade_pct / 100.0)
                    qty = risk_amount / max(sl_dist, 1e-9)

                    entry = close * (1 + spread_pct / 100.0) if action == "BUY" \
                        else close * (1 - spread_pct / 100.0)
                    sl = entry - sl_dist if action == "BUY" else entry + sl_dist
                    tp = entry + tp_dist if action == "BUY" else entry - tp_dist
                    fees = qty * entry * (commission_pct / 100.0)

                    position = {
                        "side": action,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "qty": qty,
                        "entry_time": df.index[i],
                        "fees": fees,
                    }

            equity_points.append({
                "timestamp": df.index[i].isoformat(),
                "balance": round(balance, 2),
            })

        result.final_balance = balance
        result.total_pnl = balance - start_balance
        result.equity_curve = equity_points
        result.win_rate = result.wins / result.total_trades if result.total_trades else 0.0

        # Metrics
        if pnl_series:
            pnl_arr = np.array(pnl_series)
            returns = pnl_arr / start_balance
            result.sharpe = self._sharpe(returns)
            result.sortino = self._sortino(returns)
            result.max_drawdown = self._max_drawdown(equity_points)
            gross_wins = sum(p for p in pnl_series if p > 0)
            gross_losses = abs(sum(p for p in pnl_series if p < 0))
            result.profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0

        return result

    def run_walk_forward(
        self,
        symbol: str,
        start: str,
        end: str,
        strategy_fn: Callable,
        *,
        train_days: int = 90,
        test_days: int = 30,
        timeframe: str = "1m",
        **kwargs,
    ) -> list[BacktestResult]:
        """
        Walk-forward validation: slide (train, test) windows across the period.
        Returns list of per-window BacktestResults (test-window only).
        """
        results = []
        dt_start = pd.to_datetime(start)
        dt_end = pd.to_datetime(end)
        cursor = dt_start

        while cursor + timedelta(days=train_days + test_days) <= dt_end:
            test_start = (cursor + timedelta(days=train_days)).isoformat()
            test_end = (cursor + timedelta(days=train_days + test_days)).isoformat()

            result = self.run(
                symbol,
                test_start,
                test_end,
                strategy_fn,
                timeframe=timeframe,
                **kwargs,
            )
            results.append(result)
            cursor += timedelta(days=test_days)

        return results

    # ── Metric helpers ──

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    @staticmethod
    def _compute_pnl(
        position: dict, exit_price: float, spread_pct: float, commission_pct: float
    ) -> float:
        """PnL net of spread + commission."""
        entry = position["entry"]
        qty = position["qty"]
        side = position["side"]

        # Exit with opposite spread
        exit_with_spread = (
            exit_price * (1 - spread_pct / 100.0)
            if side == "BUY"
            else exit_price * (1 + spread_pct / 100.0)
        )

        if side == "BUY":
            gross = (exit_with_spread - entry) * qty
        else:
            gross = (entry - exit_with_spread) * qty

        exit_fees = qty * exit_price * (commission_pct / 100.0)
        total_fees = position.get("fees", 0) + exit_fees
        return gross - total_fees

    @staticmethod
    def _sharpe(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - rf / periods
        std = np.std(excess)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * math.sqrt(periods))

    @staticmethod
    def _sortino(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - rf / periods
        downside = excess[excess < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess) / downside_std * math.sqrt(periods))

    @staticmethod
    def _max_drawdown(equity_curve: list[dict]) -> float:
        if not equity_curve:
            return 0.0
        balances = np.array([p["balance"] for p in equity_curve])
        running_max = np.maximum.accumulate(balances)
        drawdowns = (running_max - balances) / running_max
        return float(np.max(drawdowns))


# ── CLI ──

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Run historical backtest")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=30, help="If no start/end, use last N days")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.start:
        args.end = args.end or datetime.now().strftime("%Y-%m-%d")
        args.start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    # Simple demo strategy: buy when close > 20-period SMA
    def demo_strategy(df, i, state):
        if len(df) < 20:
            return {"action": "HOLD", "confidence": 0.0}
        sma20 = df["close"].iloc[-20:].mean()
        current = df["close"].iloc[-1]
        if current > sma20 * 1.001:
            return {"action": "BUY", "confidence": 0.7, "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}
        return {"action": "HOLD", "confidence": 0.0}

    bt = HistoricalBacktester()
    result = bt.run(args.symbol, args.start, args.end, demo_strategy, timeframe=args.timeframe)
    print(_json.dumps(result.to_dict(), indent=2))
