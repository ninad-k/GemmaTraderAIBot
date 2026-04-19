"""
Rey Capital AI Bot — Local Trader
===================================
Fetches live 1M candles from MT5 (with TradingView fallback),
calculates 30+ indicators including Ichimoku, feeds to Gemma 4,
and auto-executes trades on MetaTrader 5.

Usage:
    python local_trader.py                                    # defaults from config
    python local_trader.py --symbols US100_Spot XAUUSD_       # specific symbols
    python local_trader.py --interval 1m                      # 1-min candles
    python local_trader.py --once                             # single run
    python local_trader.py --mode paper                       # paper mode

Architecture:
    MT5 (candles) → pandas_ta (30+ indicators) → Gemma 4 (decision) → MT5 (execution)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import yaml
import schedule

from gemma_analyzer import analyze_with_gemma, SYSTEM_PROMPT
from risk_manager import RiskManager
from broker_bridge import create_broker
from symbol_registry import get_registry
from safety import get_safety, flatten_all_positions
from notifier import get_notifier
from ensemble import ensemble_decide, prompt_hash, get_dedupe_cache
from news_calendar import get_calendar

logger = logging.getLogger("rey_capital_trader")


# ─── TradingView Fallback Data Fetcher ────────────────────────

class TradingViewFallback:
    """Fallback: fetches pre-computed indicators from TradingView scanner API."""

    SCAN_URL = "https://scanner.tradingview.com/{exchange}/scan"

    # Map MT5 symbols to TradingView equivalents for fallback
    SYMBOL_MAP = {
        "US100_Spot": ("US100", "FOREXCOM"),
        "US30_SPOT": ("US30", "FOREXCOM"),
        "XAUUSD_": ("XAUUSD", "OANDA"),
        "XAGUSD_": ("XAGUSD", "OANDA"),
        "BTCUSD": ("BTCUSD", "COINBASE"),
    }

    EXCHANGE_TO_SCANNER = {
        "FOREXCOM": "cfd", "OANDA": "forex",
        "COINBASE": "crypto", "BINANCE": "crypto",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Origin": "https://www.tradingview.com",
        })

    def fetch_indicators(self, mt5_symbol: str) -> dict:
        """Fetch pre-computed indicators from TradingView for a given MT5 symbol."""
        mapping = self.SYMBOL_MAP.get(mt5_symbol)
        if not mapping:
            logger.warning(f"No TradingView mapping for {mt5_symbol}")
            return {}

        tv_symbol, exchange = mapping
        scanner = self.EXCHANGE_TO_SCANNER.get(exchange, "america")

        columns = [
            "open", "high", "low", "close", "volume",
            "RSI", "MACD.macd", "MACD.signal", "MACD.hist",
            "EMA20", "EMA50", "EMA200",
            "ATR", "BB.upper", "BB.lower",
            "Stoch.K", "Stoch.D", "ADX", "ADX+DI", "ADX-DI",
            "VWAP", "Recommend.All",
            "Ichimoku.BLine", "Ichimoku.CLine",
            "CCI20", "W.R", "Mom",
        ]

        payload = {
            "symbols": {"tickers": [f"{exchange}:{tv_symbol}"]},
            "columns": columns,
        }

        try:
            url = self.SCAN_URL.format(exchange=scanner)
            resp = self.session.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("data"):
                return {}

            values = data["data"][0]["d"]
            raw = dict(zip(columns, values))

            def sf(key, default=0):
                v = raw.get(key)
                return round(float(v), 4) if v is not None else default

            ema20 = sf("EMA20")
            ema50 = sf("EMA50")

            return {
                "symbol": mt5_symbol,
                "close": sf("close"), "open": sf("open"),
                "high": sf("high"), "low": sf("low"),
                "volume": sf("volume"),
                "rsi": sf("RSI", 50),
                "macd": sf("MACD.macd"), "macd_signal": sf("MACD.signal"),
                "macd_hist": sf("MACD.hist"),
                "ema9": 0, "ema20": ema20, "ema50": ema50,
                "ema200": sf("EMA200"), "sma20": 0, "sma50": 0,
                "atr": sf("ATR"),
                "bb_upper": sf("BB.upper"), "bb_lower": sf("BB.lower"),
                "bb_mid": (sf("BB.upper") + sf("BB.lower")) / 2,
                "stoch_k": sf("Stoch.K", 50), "stoch_d": sf("Stoch.D", 50),
                "adx": sf("ADX"), "di_plus": sf("ADX+DI"), "di_minus": sf("ADX-DI"),
                "vwap": sf("VWAP", sf("close")),
                "cci": sf("CCI20"), "williams_r": sf("W.R"),
                "ichimoku_tenkan": sf("Ichimoku.CLine"),
                "ichimoku_kijun": sf("Ichimoku.BLine"),
                "trend": "BULLISH" if ema20 > ema50 else "BEARISH",
                "data_source": "tradingview_fallback",
            }
        except Exception as e:
            logger.error(f"TradingView fallback failed for {mt5_symbol}: {e}")
            return {}


# ─── Comprehensive Indicator Calculator ───────────────────────

def calculate_indicators(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Calculate 30+ technical indicators from OHLCV data.
    Includes Ichimoku, Supertrend, CCI, Williams %R, OBV, and more.
    Returns a dict ready for Gemma 4 analysis.
    """
    if df is None or len(df) < 50:
        logger.warning(f"Not enough data: {len(df) if df is not None else 0} bars (need 50+)")
        return {}

    # Make a copy to avoid modifying original
    df = df.copy()

    # ── Trend Indicators ──
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.adx(length=14, append=True)

    # Ichimoku Cloud
    df.ta.ichimoku(append=True)

    # Supertrend
    try:
        df.ta.supertrend(length=10, multiplier=3.0, append=True)
    except Exception:
        pass

    # Parabolic SAR
    try:
        df.ta.psar(append=True)
    except Exception:
        pass

    # VWAP (may fail without datetime index)
    try:
        df.ta.vwap(append=True)
    except Exception:
        pass

    # ── Momentum Indicators ──
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.stochrsi(length=14, append=True)
    df.ta.stoch(append=True)
    df.ta.cci(length=20, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.roc(length=10, append=True)
    df.ta.mfi(length=14, append=True)

    # ── Volatility Indicators ──
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2.0, append=True)

    # Keltner Channels
    try:
        df.ta.kc(length=20, scalar=1.5, append=True)
    except Exception:
        pass

    # Donchian Channels
    try:
        df.ta.donchian(lower_length=20, upper_length=20, append=True)
    except Exception:
        pass

    # ── Volume Indicators ──
    try:
        df.ta.obv(append=True)
    except Exception:
        pass

    try:
        df.ta.ad(append=True)
    except Exception:
        pass

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    # ── Helper: safe float get ──
    def sf(col, default=0):
        try:
            v = latest.get(col, default)
            if pd.isna(v):
                return default
            return round(float(v), 4)
        except (TypeError, ValueError):
            return default

    def sf_prev(col, default=0):
        try:
            v = prev.get(col, default)
            if pd.isna(v):
                return default
            return round(float(v), 4)
        except (TypeError, ValueError):
            return default

    # ── Trend Analysis ──
    ema9 = sf("EMA_9")
    ema20 = sf("EMA_20")
    ema50 = sf("EMA_50")
    ema200 = sf("EMA_200")

    # Multi-EMA trend strength
    if ema9 > ema20 > ema50 > ema200:
        trend = "STRONG_BULLISH"
    elif ema9 > ema20 > ema50:
        trend = "BULLISH"
    elif ema9 < ema20 < ema50 < ema200:
        trend = "STRONG_BEARISH"
    elif ema9 < ema20 < ema50:
        trend = "BEARISH"
    else:
        trend = "MIXED"

    # EMA crossover detection
    prev_ema9 = sf_prev("EMA_9")
    prev_ema20 = sf_prev("EMA_20")
    if prev_ema9 <= prev_ema20 and ema9 > ema20:
        ema_cross = "BULLISH_CROSS_9_20"
    elif prev_ema9 >= prev_ema20 and ema9 < ema20:
        ema_cross = "BEARISH_CROSS_9_20"
    else:
        prev_ema20_2 = sf_prev("EMA_20")
        prev_ema50_2 = sf_prev("EMA_50")
        if prev_ema20_2 <= prev_ema50_2 and ema20 > ema50:
            ema_cross = "GOLDEN_CROSS"
        elif prev_ema20_2 >= prev_ema50_2 and ema20 < ema50:
            ema_cross = "DEATH_CROSS"
        else:
            ema_cross = "NONE"

    # ── Volume Analysis ──
    vol_sma = df["volume"].rolling(20).mean().iloc[-1]
    vol_ratio = round(float(latest["volume"] / vol_sma), 2) if vol_sma > 0 else 1.0
    vol_trend = "SURGE" if vol_ratio > 2.0 else "HIGH" if vol_ratio > 1.5 else \
                "ABOVE_AVG" if vol_ratio > 1.0 else "LOW"

    # ── Bollinger Band Position ──
    bb_upper = sf("BBU_20_2.0")
    bb_lower = sf("BBL_20_2.0")
    bb_mid = sf("BBM_20_2.0")
    close = sf("close", float(latest["close"]))
    bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2) if bb_mid > 0 else 0

    if close > bb_upper:
        bb_pos = "ABOVE_UPPER"
    elif close < bb_lower:
        bb_pos = "BELOW_LOWER"
    elif close > bb_mid:
        bb_pos = "UPPER_HALF"
    else:
        bb_pos = "LOWER_HALF"

    # ── Ichimoku Cloud Analysis ──
    isa = sf("ISA_9")  # Senkou Span A
    isb = sf("ISB_26")  # Senkou Span B
    its = sf("ITS_9")   # Tenkan-sen
    iks = sf("IKS_26")  # Kijun-sen

    if close > max(isa, isb) and its > iks:
        ichimoku_signal = "STRONG_BULLISH"
    elif close > max(isa, isb):
        ichimoku_signal = "BULLISH"
    elif close < min(isa, isb) and its < iks:
        ichimoku_signal = "STRONG_BEARISH"
    elif close < min(isa, isb):
        ichimoku_signal = "BEARISH"
    else:
        ichimoku_signal = "IN_CLOUD"

    cloud_color = "GREEN" if isa > isb else "RED"

    # ── Supertrend ──
    st_col = [c for c in df.columns if c.startswith("SUPERT_")]
    supertrend_dir = "NONE"
    if st_col:
        st_val = sf(st_col[0])
        supertrend_dir = "BULLISH" if close > st_val else "BEARISH"

    # ── Parabolic SAR ──
    psar_long = sf("PSARl_0.02_0.2")
    psar_short = sf("PSARs_0.02_0.2")
    psar_signal = "BULLISH" if psar_long > 0 and close > psar_long else \
                  "BEARISH" if psar_short > 0 and close < psar_short else "NEUTRAL"

    # ── Candlestick Pattern Detection ──
    patterns = _detect_candle_patterns(df)

    # ── Last 5 Candles for Price Action ──
    last5 = []
    for i in range(-5, 0):
        if abs(i) <= len(df):
            c = df.iloc[i]
            body = abs(float(c["close"]) - float(c["open"]))
            total = float(c["high"]) - float(c["low"])
            body_pct = round(body / total * 100, 1) if total > 0 else 0
            direction = "UP" if float(c["close"]) >= float(c["open"]) else "DOWN"
            last5.append(f"{direction}({body_pct}%)")

    # ── Support/Resistance from recent swing points ──
    support, resistance = _find_support_resistance(df)

    # ── Build comprehensive indicator dict ──
    return {
        "symbol": symbol,
        "data_source": "mt5",
        "close": close,
        "open": sf("open", float(latest["open"])),
        "high": sf("high", float(latest["high"])),
        "low": sf("low", float(latest["low"])),
        "volume": round(float(latest["volume"]), 0),

        # Trend
        "ema9": ema9, "ema20": ema20, "ema50": ema50, "ema200": ema200,
        "sma20": sf("SMA_20"), "sma50": sf("SMA_50"),
        "trend": trend,
        "ema_cross": ema_cross,
        "adx": sf("ADX_14"), "di_plus": sf("DMP_14"), "di_minus": sf("DMN_14"),
        "supertrend": supertrend_dir,
        "psar_signal": psar_signal,

        # Ichimoku
        "ichimoku_tenkan": its, "ichimoku_kijun": iks,
        "ichimoku_span_a": isa, "ichimoku_span_b": isb,
        "ichimoku_signal": ichimoku_signal,
        "ichimoku_cloud_color": cloud_color,

        # Momentum
        "rsi": sf("RSI_14", 50),
        "macd": sf("MACD_12_26_9"), "macd_signal": sf("MACDs_12_26_9"),
        "macd_hist": sf("MACDh_12_26_9"),
        "stoch_rsi_k": sf("STOCHRSIk_14_14_3_3", 50),
        "stoch_rsi_d": sf("STOCHRSId_14_14_3_3", 50),
        "stoch_k": sf("STOCHk_14_3_3", 50),
        "stoch_d": sf("STOCHd_14_3_3", 50),
        "cci": sf("CCI_20_0.015"),
        "williams_r": sf("WILLR_14"),
        "roc": sf("ROC_10"),
        "mfi": sf("MFI_14", 50),

        # Volatility
        "atr": sf("ATRr_14"),
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "bb_pos": bb_pos, "bb_width": bb_width,

        # Volume
        "vol_trend": vol_trend, "vol_ratio": vol_ratio,
        "obv": sf("OBV"),

        # Candlestick Patterns
        "candle_patterns": ", ".join(patterns) if patterns else "NONE",
        "last_5_candles": " → ".join(last5),

        # Support/Resistance
        "nearest_support": support,
        "nearest_resistance": resistance,

        # VWAP
        "vwap": sf("VWAP_D", close),
    }


def _detect_candle_patterns(df: pd.DataFrame) -> list:
    """Detect candlestick patterns from the last few candles."""
    patterns = []
    if len(df) < 3:
        return patterns

    c = df.iloc[-1]
    p = df.iloc[-2]
    pp = df.iloc[-3]

    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    po, ph, pl, pcl = float(p["open"]), float(p["high"]), float(p["low"]), float(p["close"])

    body = abs(cl - o)
    total_range = h - l
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l

    if total_range == 0:
        return patterns

    body_pct = body / total_range

    # Doji
    if body_pct < 0.1:
        patterns.append("DOJI")

    # Hammer (bullish reversal)
    if lower_wick > body * 2 and upper_wick < body * 0.5 and cl > o:
        patterns.append("HAMMER")

    # Shooting Star (bearish reversal)
    if upper_wick > body * 2 and lower_wick < body * 0.5 and cl < o:
        patterns.append("SHOOTING_STAR")

    # Bullish Engulfing
    if pcl < po and cl > o and cl > po and o < pcl:
        patterns.append("BULLISH_ENGULFING")

    # Bearish Engulfing
    if pcl > po and cl < o and cl < po and o > pcl:
        patterns.append("BEARISH_ENGULFING")

    # Morning Star (3 candle bullish)
    if len(df) >= 3:
        ppo, ppcl = float(pp["open"]), float(pp["close"])
        if ppcl < ppo and abs(pcl - po) < (ph - pl) * 0.3 and cl > o and cl > (ppo + ppcl) / 2:
            patterns.append("MORNING_STAR")

    # Evening Star (3 candle bearish)
    if len(df) >= 3:
        ppo, ppcl = float(pp["open"]), float(pp["close"])
        if ppcl > ppo and abs(pcl - po) < (ph - pl) * 0.3 and cl < o and cl < (ppo + ppcl) / 2:
            patterns.append("EVENING_STAR")

    # Three White Soldiers
    if len(df) >= 3:
        ppo, ppcl = float(pp["open"]), float(pp["close"])
        if ppcl > ppo and pcl > po and cl > o and cl > pcl > ppcl:
            patterns.append("THREE_WHITE_SOLDIERS")

    # Three Black Crows
    if len(df) >= 3:
        ppo, ppcl = float(pp["open"]), float(pp["close"])
        if ppcl < ppo and pcl < po and cl < o and cl < pcl < ppcl:
            patterns.append("THREE_BLACK_CROWS")

    return patterns


def _find_support_resistance(df: pd.DataFrame, lookback: int = 50) -> tuple:
    """Find nearest support and resistance from swing highs/lows."""
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)
    close = float(df.iloc[-1]["close"])

    # Find swing lows (support) and swing highs (resistance)
    highs = recent["high"].astype(float).values
    lows = recent["low"].astype(float).values

    supports = []
    resistances = []

    for i in range(2, len(highs) - 2):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            if highs[i] > close:
                resistances.append(highs[i])
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            if lows[i] < close:
                supports.append(lows[i])

    nearest_support = round(max(supports), 2) if supports else round(close * 0.998, 2)
    nearest_resistance = round(min(resistances), 2) if resistances else round(close * 1.002, 2)

    return nearest_support, nearest_resistance


# ─── Main Trading Loop ────────────────────────────────────────

class GemmaLocalTrader:
    """
    Rey Capital AI Bot — Main trading engine.
    Pulls data from MT5 (or TradingView fallback), analyzes with Gemma 4,
    and auto-executes trades on MetaTrader 5.
    """

    def __init__(self, config: dict, symbols: list = None, interval: str = None):
        self.config = config
        self.interval = interval or config.get("mt5_data", {}).get("timeframe", "1m")
        self.registry = get_registry()
        if symbols:
            self.symbols = symbols
        else:
            active = self.registry.active_generics()
            self.symbols = active or config["trading"]["allowed_symbols"]
        self.risk_manager = RiskManager(config)
        self.broker = create_broker(config)
        self.cycle_count = 0
        self.socketio = None  # Set externally by run.py for WebSocket
        self.safety = get_safety(config)
        self.notifier = get_notifier()
        self.safety.attach_notifier(self.notifier)
        self.news = get_calendar()
        self.dedupe = get_dedupe_cache(ttl_seconds=int(config.get("ensemble", {}).get("dedupe_ttl", 60)))

        # Initialize data sources
        self.use_mt5 = config.get("data_source", "mt5") == "mt5"
        self.mt5_feed = None
        self.tv_fallback = TradingViewFallback()

        if self.use_mt5:
            try:
                from mt5_data_feed import MT5DataFeed
                self.mt5_feed = MT5DataFeed(config)
                if not self.mt5_feed.connected:
                    logger.warning("MT5 not connected, will use TradingView fallback")
                    self.mt5_feed = None
            except ImportError:
                logger.warning("MT5 package not available, using TradingView fallback")
            except Exception as e:
                logger.warning(f"MT5 init failed: {e}, using TradingView fallback")

    def analyze_symbol(self, symbol: str):
        """Full pipeline: fetch data → calculate indicators → Gemma decision → execute trade."""
        logger.info(f"\n{'='*55}")
        logger.info(f"  Analyzing {symbol} on {self.interval}")
        logger.info(f"{'='*55}")

        indicators = None

        # ── Method 1: MT5 Data Feed (primary) ──
        if self.mt5_feed and self.mt5_feed.connected:
            try:
                n_bars = self.config.get("mt5_data", {}).get("n_bars", 500)
                df = self.mt5_feed.get_candles(symbol, self.interval, n_bars)
                if df is not None and len(df) >= 50:
                    logger.info(f"  MT5: {len(df)} candles | Close: {df['close'].iloc[-1]:.2f}")
                    indicators = calculate_indicators(df, symbol)
                    if indicators:
                        indicators["timeframe"] = self.interval
                else:
                    logger.warning(f"  MT5: insufficient data ({len(df) if df is not None else 0} bars)")
            except Exception as e:
                logger.warning(f"  MT5 data fetch failed: {e}")

        # ── Method 2: TradingView Fallback ──
        if not indicators:
            logger.info("  Using TradingView fallback...")
            try:
                indicators = self.tv_fallback.fetch_indicators(symbol)
                if indicators:
                    indicators["timeframe"] = self.interval
                    logger.info(f"  TV fallback: Close: {indicators.get('close', '?')}")
            except Exception as e:
                logger.error(f"  TradingView fallback also failed: {e}")
                return

        if not indicators:
            logger.error(f"  No data available for {symbol}")
            return

        # Log key indicators
        logger.info(
            f"  RSI: {indicators.get('rsi', '?')} | "
            f"MACD: {indicators.get('macd_hist', '?')} | "
            f"Trend: {indicators.get('trend', '?')} | "
            f"Ichimoku: {indicators.get('ichimoku_signal', '?')} | "
            f"Vol: {indicators.get('vol_trend', '?')}"
        )

        # ── Safety gate: halt blocks all new entries ──
        if self.safety.is_halted():
            logger.warning(f"  Skipping {symbol}: trading halted ({self.safety.state.halt_reason})")
            return

        # ── News blackout ──
        in_blackout, label = self.news.in_blackout()
        if in_blackout:
            logger.info(f"  Skipping {symbol}: news blackout ({label})")
            return

        # ── Ask Gemma (with dedupe cache + optional ensemble) ──
        cached = self.dedupe.lookup(indicators)
        if cached is not None:
            decision = cached
            logger.info(f"  [dedupe] reusing cached decision for {symbol}")
        else:
            decision = ensemble_decide(indicators, self.config, analyze_with_gemma)
            self.dedupe.store(indicators, decision)
        decision["prompt_hash"] = prompt_hash(SYSTEM_PROMPT, decision.get("prompt_sent", ""))
        logger.info(
            f"  Gemma: {decision['action']} "
            f"({decision['confidence']:.0%}) — {decision['reason']}"
        )

        # Log decision for dashboard
        self._log_decision(symbol, indicators.get("close", 0), decision, indicators)

        # Emit WebSocket event if available
        if self.socketio:
            self.socketio.emit("new_decision", {
                "symbol": symbol,
                "close": indicators.get("close", 0),
                "decision": decision,
                "timestamp": datetime.now().isoformat(),
            })

        # ── Risk Check ──
        if decision["action"] == "HOLD":
            return

        allowed, reason = self.risk_manager.can_trade(decision, indicators)
        if not allowed:
            logger.warning(f"  Blocked: {reason}")
            return

        # ── Position Sizing ──
        balance = self.broker.get_balance()
        atr = float(indicators.get("atr", 0))
        if atr <= 0:
            logger.warning("  ATR is zero, cannot calculate position size")
            return

        # Check for manual lot size override from dashboard
        lot_override = self._get_lot_override(symbol)
        if lot_override and lot_override > 0:
            pos_size = {
                "qty": lot_override,
                "risk_amount": 0,
                "sl_distance": atr * decision.get("sl_distance_atr", 1.0),
                "override": True,
            }
            logger.info(f"  LOT OVERRIDE: Using manual lot size {lot_override} for {symbol}")
        else:
            pos_size = self.risk_manager.calculate_position_size(
                balance, atr, decision.get("sl_distance_atr", 1.0), symbol
            )

        if pos_size["qty"] <= 0:
            logger.warning("  Invalid position size")
            return

        # Log risk calculation details
        risk_pct = self.config["trading"]["max_position_size_pct"]
        logger.info(
            f"  RISK: {risk_pct}% of ${balance:,.2f} = ${pos_size.get('risk_amount', 0):.2f} risk | "
            f"ATR={atr:.5f} | SL dist={pos_size.get('sl_distance', 0):.5f} | "
            f"Lots={pos_size['qty']} {'(OVERRIDE)' if pos_size.get('override') else ''}"
        )

        # ── Calculate SL/TP Prices ──
        # Use actual bid/ask from MT5 for accurate SL/TP placement
        close = float(indicators["close"])
        sl_dist = atr * decision.get("sl_distance_atr", 1.0)
        tp_dist = atr * decision.get("tp_distance_atr", 1.5)

        # Get real-time tick for spread-aware SL/TP
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
            sym_info = mt5.symbol_info(symbol)
            if tick and sym_info:
                spread = tick.ask - tick.bid
                digits = sym_info.digits
                freeze = sym_info.trade_freeze_level * sym_info.point
                # Ensure SL/TP are at least spread + freeze beyond entry
                min_dist = spread + freeze + sym_info.point * 10
                sl_dist = max(sl_dist, min_dist)
                tp_dist = max(tp_dist, min_dist)
                if decision["action"] == "BUY":
                    entry = tick.ask
                    sl = round(entry - sl_dist, digits)
                    tp = round(entry + tp_dist, digits)
                else:
                    entry = tick.bid
                    sl = round(entry + sl_dist, digits)
                    tp = round(entry - tp_dist, digits)
                logger.info(f"  SL/TP calc: entry={entry} spread={spread:.{digits}f} atr={atr:.{digits}f} sl_dist={sl_dist:.{digits}f} tp_dist={tp_dist:.{digits}f}")
            else:
                raise ValueError("No tick data")
        except Exception as e:
            logger.warning(f"  Could not get tick for SL/TP, using close: {e}")
            if decision["action"] == "BUY":
                sl = round(close - sl_dist, 5)
                tp = round(close + tp_dist, 5)
            else:
                sl = round(close + sl_dist, 5)
                tp = round(close - tp_dist, 5)

        # ── Execute Trade ──
        logger.info(
            f"  EXECUTING {decision['action']} {symbol} | "
            f"Qty: {pos_size['qty']} | SL: {sl} | TP: {tp}"
        )

        result = self.broker.place_order(
            symbol, decision["action"], pos_size["qty"], sl, tp
        )

        trade_data = {
            "symbol": symbol,
            "action": decision["action"],
            "qty": pos_size["qty"],
            "entry_price": close,
            "sl": sl,
            "tp": tp,
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "indicators_snapshot": {
                "rsi": indicators.get("rsi"),
                "macd_hist": indicators.get("macd_hist"),
                "trend": indicators.get("trend"),
                "ichimoku_signal": indicators.get("ichimoku_signal"),
                "adx": indicators.get("adx"),
                "vol_trend": indicators.get("vol_trend"),
                "candle_patterns": indicators.get("candle_patterns"),
            },
            "timestamp": datetime.now().isoformat(),
        }

        if result.get("status") == "filled":
            self.risk_manager.register_trade(trade_data)
            logger.info(f"  Trade FILLED!")

            # Write trade journal entry with full Gemma thinking
            self._write_journal_entry(trade_data, decision, indicators, pos_size, balance)

            try:
                self.notifier.notify(
                    "entry",
                    f"{decision['action']} {symbol} qty={pos_size['qty']} @ {close}",
                    {"sl": sl, "tp": tp, "conf": f"{decision['confidence']:.2f}"},
                )
            except Exception as e:
                logger.warning(f"entry notify failed: {e}")

            if self.socketio:
                self.socketio.emit("new_trade", trade_data)
        else:
            logger.error(f"  Trade FAILED: {result}")
            # Still journal the failed attempt
            trade_data["status"] = "FAILED"
            trade_data["error"] = result.get("comment", str(result))
            self._write_journal_entry(trade_data, decision, indicators, pos_size, balance)

    def _get_lot_override(self, symbol: str) -> float:
        """Check if there's a manual lot size override for this symbol."""
        try:
            override_path = Path("logs/lot_overrides.json")
            if override_path.exists():
                text = override_path.read_text(encoding="utf-8-sig").strip()
                if text:
                    overrides = json.loads(text)
                    return overrides.get(symbol, 0)
        except Exception:
            pass
        return 0

    def _write_journal_entry(self, trade_data: dict, decision: dict,
                             indicators: dict, pos_size: dict, balance: float):
        """Write a comprehensive trade journal entry with Gemma's full thinking."""
        journal_entry = {
            "trade_id": f"T-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{trade_data['symbol']}",
            "timestamp": datetime.now().isoformat(),
            "symbol": trade_data["symbol"],
            "action": trade_data["action"],
            "status": trade_data.get("status", "FILLED"),

            # Position details
            "lot_size": trade_data["qty"],
            "entry_price": trade_data.get("entry_price"),
            "sl": trade_data.get("sl"),
            "tp": trade_data.get("tp"),
            "lot_override": pos_size.get("override", False),

            # Risk calculation
            "risk_calculation": {
                "balance": balance,
                "risk_pct": self.config["trading"]["max_position_size_pct"],
                "risk_amount": pos_size.get("risk_amount", 0),
                "atr": float(indicators.get("atr", 0)),
                "sl_distance": pos_size.get("sl_distance", 0),
                "sl_atr_mult": decision.get("sl_distance_atr", 1.0),
                "tp_atr_mult": decision.get("tp_distance_atr", 1.5),
            },

            # Gemma's analysis
            "gemma_decision": {
                "action": decision.get("action"),
                "confidence": decision.get("confidence"),
                "reason": decision.get("reason"),
                "sl_distance_atr": decision.get("sl_distance_atr"),
                "tp_distance_atr": decision.get("tp_distance_atr"),
            },
            "gemma_raw_thinking": decision.get("raw_gemma_response", ""),
            "prompt_sent": decision.get("prompt_sent", ""),

            # Indicators at entry
            "indicators_at_entry": {
                "close": indicators.get("close"),
                "rsi": indicators.get("rsi"),
                "macd": indicators.get("macd"),
                "macd_hist": indicators.get("macd_hist"),
                "trend": indicators.get("trend"),
                "ema9": indicators.get("ema9"),
                "ema20": indicators.get("ema20"),
                "ema50": indicators.get("ema50"),
                "ema200": indicators.get("ema200"),
                "ichimoku_signal": indicators.get("ichimoku_signal"),
                "ichimoku_cloud_color": indicators.get("ichimoku_cloud_color"),
                "supertrend": indicators.get("supertrend"),
                "psar_signal": indicators.get("psar_signal"),
                "adx": indicators.get("adx"),
                "di_plus": indicators.get("di_plus"),
                "di_minus": indicators.get("di_minus"),
                "bb_pos": indicators.get("bb_pos"),
                "bb_width": indicators.get("bb_width"),
                "stoch_rsi_k": indicators.get("stoch_rsi_k"),
                "stoch_k": indicators.get("stoch_k"),
                "cci": indicators.get("cci"),
                "williams_r": indicators.get("williams_r"),
                "mfi": indicators.get("mfi"),
                "roc": indicators.get("roc"),
                "vol_trend": indicators.get("vol_trend"),
                "vol_ratio": indicators.get("vol_ratio"),
                "vwap": indicators.get("vwap"),
                "candle_patterns": indicators.get("candle_patterns"),
                "nearest_support": indicators.get("nearest_support"),
                "nearest_resistance": indicators.get("nearest_resistance"),
            },

            # Strategy classification
            "strategy_signals": self._classify_strategy(decision, indicators),

            # User comments (empty initially, added via dashboard)
            "comments": [],
            "error": trade_data.get("error", ""),
        }

        try:
            journal_path = Path("logs/trade_journal.json")
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            entries = []
            if journal_path.exists():
                text = journal_path.read_text(encoding="utf-8-sig").strip()
                if text:
                    entries = json.loads(text)
            entries.append(journal_entry)
            journal_path.write_text(json.dumps(entries[-500:], indent=2))
            logger.info(f"  Journal entry written: {journal_entry['trade_id']}")
        except Exception as e:
            logger.warning(f"Could not write journal: {e}")

    def _classify_strategy(self, decision: dict, indicators: dict) -> dict:
        """Classify which strategies/signals contributed to the trade."""
        signals = []
        reasons = []

        # Trend alignment
        trend = indicators.get("trend", "")
        action = decision.get("action", "")
        if "BULLISH" in str(trend) and action == "BUY":
            signals.append("TREND_ALIGNED")
            reasons.append(f"Trend is {trend}, aligned with BUY")
        elif "BEARISH" in str(trend) and action == "SELL":
            signals.append("TREND_ALIGNED")
            reasons.append(f"Trend is {trend}, aligned with SELL")

        # Ichimoku
        ichi = indicators.get("ichimoku_signal", "")
        if ichi:
            signals.append(f"ICHIMOKU_{ichi}")
            reasons.append(f"Ichimoku Cloud: {ichi}")

        # Supertrend
        st = indicators.get("supertrend", "")
        if st:
            signals.append(f"SUPERTREND_{st}")
            reasons.append(f"Supertrend: {st}")

        # RSI zones
        rsi = indicators.get("rsi")
        if rsi:
            rsi = float(rsi)
            if rsi < 30:
                signals.append("RSI_OVERSOLD")
                reasons.append(f"RSI oversold at {rsi:.1f}")
            elif rsi > 70:
                signals.append("RSI_OVERBOUGHT")
                reasons.append(f"RSI overbought at {rsi:.1f}")

        # MACD momentum
        macd_hist = indicators.get("macd_hist")
        if macd_hist:
            macd_hist = float(macd_hist)
            if macd_hist > 0 and action == "BUY":
                signals.append("MACD_BULLISH")
                reasons.append(f"MACD histogram positive ({macd_hist:.4f})")
            elif macd_hist < 0 and action == "SELL":
                signals.append("MACD_BEARISH")
                reasons.append(f"MACD histogram negative ({macd_hist:.4f})")

        # Volume confirmation
        vol = indicators.get("vol_trend", "")
        if "HIGH" in str(vol) or "ABOVE" in str(vol):
            signals.append("VOLUME_CONFIRMED")
            reasons.append(f"Volume: {vol} ({indicators.get('vol_ratio', '?')}x avg)")

        # Candlestick patterns
        patterns = indicators.get("candle_patterns", "")
        if patterns and patterns != "NONE":
            signals.append(f"CANDLE_PATTERN")
            reasons.append(f"Patterns: {patterns}")

        # ADX strength
        adx = indicators.get("adx")
        if adx:
            adx = float(adx)
            if adx > 25:
                signals.append("STRONG_TREND")
                reasons.append(f"ADX strong trend at {adx:.1f}")

        return {
            "signals": signals,
            "signal_count": len(signals),
            "confluence_score": min(len(signals) / 5.0, 1.0),
            "reasons": reasons,
            "core_strategy": self._determine_core_strategy(signals),
        }

    def _determine_core_strategy(self, signals: list) -> str:
        """Determine the primary strategy used for this trade."""
        if "TREND_ALIGNED" in signals and "ICHIMOKU_STRONG_BULLISH" in signals or "ICHIMOKU_STRONG_BEARISH" in signals:
            return "Ichimoku Cloud Trend Following"
        if "SUPERTREND_BUY" in signals or "SUPERTREND_SELL" in signals:
            return "Supertrend Momentum"
        if "RSI_OVERSOLD" in signals or "RSI_OVERBOUGHT" in signals:
            return "RSI Mean Reversion"
        if "MACD_BULLISH" in signals or "MACD_BEARISH" in signals:
            return "MACD Momentum"
        if "VOLUME_CONFIRMED" in signals:
            return "Volume Breakout"
        return "Multi-Indicator Confluence"

    def _check_closed_positions(self):
        """Check if any tracked positions were closed by SL/TP on MT5."""
        if not self.mt5_feed or not self.mt5_feed.connected:
            return

        open_positions = self.mt5_feed.get_positions()
        open_symbols = {p["symbol"] for p in open_positions if p.get("magic") == 240411}

        for trade in list(self.risk_manager.open_trades):
            if trade["symbol"] not in open_symbols:
                # Position was closed (hit SL or TP)
                logger.info(f"  Position closed: {trade['symbol']}")

                # Try to get actual PnL from MT5 deal history
                profit = 0.0
                close_price = trade.get("entry_price", 0)
                try:
                    deals = self.mt5_feed.get_deals_history(days=1)
                    for deal in reversed(deals):
                        if deal["symbol"] == trade["symbol"]:
                            profit = deal.get("profit", 0)
                            close_price = deal.get("price", close_price)
                            break
                except Exception as e:
                    logger.warning(f"Could not fetch deal history: {e}")

                # Record outcome for self-learning
                self.risk_manager.record_outcome(trade, close_price, profit)
                self.risk_manager.close_trade(trade["symbol"], profit)

                try:
                    self.notifier.notify(
                        "exit",
                        f"Closed {trade['symbol']} profit={profit:.2f}",
                        {"action": trade.get("action"), "qty": trade.get("qty")},
                    )
                except Exception as e:
                    logger.warning(f"exit notify failed: {e}")

                if self.socketio:
                    self.socketio.emit("trade_closed", {
                        "symbol": trade["symbol"],
                        "profit": profit,
                        "timestamp": datetime.now().isoformat(),
                    })

    def _log_decision(self, symbol: str, close: float, decision: dict,
                      indicators: dict = None):
        """Log Gemma's decision for the dashboard with full thinking."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "close": close,
            "decision": {
                "action": decision.get("action"),
                "confidence": decision.get("confidence"),
                "reason": decision.get("reason"),
                "sl_distance_atr": decision.get("sl_distance_atr"),
                "tp_distance_atr": decision.get("tp_distance_atr"),
            },
            "raw_gemma_response": decision.get("raw_gemma_response", ""),
            "indicators_summary": decision.get("indicators_summary", {}),
        }
        try:
            log_path = Path("logs/gemma_decisions.json")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logs = []
            if log_path.exists():
                text = log_path.read_text(encoding="utf-8-sig").strip()
                if text:
                    logs = json.loads(text)
            logs.append(entry)
            log_path.write_text(json.dumps(logs[-2000:], indent=2))
        except Exception as e:
            logger.warning(f"Could not write decision log: {e}")

    def _reconnect_mt5_if_needed(self) -> None:
        """If MT5 feed/broker dropped, try to re-init; notify on transition."""
        feed_ok = bool(self.mt5_feed and self.mt5_feed.connected)
        if self.use_mt5 and not feed_ok:
            try:
                from mt5_data_feed import MT5DataFeed
                self.mt5_feed = MT5DataFeed(self.config)
                if self.mt5_feed.connected:
                    logger.info("MT5 reconnected")
                    try:
                        self.notifier.notify("reconnect", "MT5 reconnected")
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"MT5 reconnect attempt failed: {e}")

    def run_cycle(self):
        """Run one analysis cycle for all symbols."""
        self.cycle_count += 1
        self.safety.heartbeat()
        self._reconnect_mt5_if_needed()
        balance = self.broker.get_balance()
        try:
            self.safety.update_equity(balance)
        except Exception:
            pass

        logger.info(f"\n{'#'*60}")
        logger.info(f"  REY CAPITAL AI BOT — Cycle #{self.cycle_count}")
        logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Balance: {balance:,.2f} | Open: {len(self.risk_manager.open_trades)} | "
                     f"Daily PnL: {self.risk_manager.daily_pnl:.2f}%")
        logger.info(f"{'#'*60}")

        # Check if any positions were closed by SL/TP
        self._check_closed_positions()

        # Refresh active symbol list from the registry so settings-UI
        # changes take effect without a restart.
        try:
            self.registry.load()
            live = self.registry.active_generics()
            if live:
                self.symbols = live
        except Exception:
            pass

        # Analyze each symbol
        for symbol in self.symbols:
            try:
                self.analyze_symbol(symbol)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            time.sleep(1)  # Brief pause between symbols

        # Emit stats update
        if self.socketio:
            self.socketio.emit("stats_update", {
                "cycle": self.cycle_count,
                "balance": balance,
                "open_trades": len(self.risk_manager.open_trades),
                "daily_pnl": self.risk_manager.daily_pnl,
                "timestamp": datetime.now().isoformat(),
            })

    def start(self, run_every_seconds: int = None):
        """Start the trading bot with recurring schedule."""
        if run_every_seconds is None:
            run_every_seconds = self.config.get("mt5_data", {}).get(
                "poll_interval_seconds", 60
            )

        logger.info(f"""
╔═══════════════════════════════════════════════════════════╗
║            REY CAPITAL AI BOT                             ║
╠═══════════════════════════════════════════════════════════╣
║  Mode:      {self.config['trading']['mode'].upper():<46}║
║  Model:     {self.config['ollama']['model']:<46}║
║  Symbols:   {', '.join(self.symbols):<46}║
║  Interval:  {self.interval:<46}║
║  Schedule:  Every {run_every_seconds}s{' '*(42-len(str(run_every_seconds)))}║
║  Data:      {'MT5' if self.mt5_feed else 'TradingView Fallback':<46}║
║  Broker:    {self.broker.__class__.__name__:<46}║
╚═══════════════════════════════════════════════════════════╝
        """)

        # Run first cycle immediately
        self.run_cycle()

        # Schedule recurring runs
        schedule.every(run_every_seconds).seconds.do(self.run_cycle)

        logger.info(f"\n  Next cycle in {run_every_seconds}s. Press Ctrl+C to stop.\n")

        try:
            while True:
                schedule.run_pending()
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("\n  Bot stopped by user")
            if self.mt5_feed:
                self.mt5_feed.shutdown()


# ─── Setup & CLI ──────────────────────────────────────────────

def setup_logging(level: str = "INFO"):
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/local_trader.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Rey Capital AI Bot")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+", help="Override symbols")
    parser.add_argument("--interval", default=None, help="Candle interval (default from config)")
    parser.add_argument("--every", type=int, help="Poll interval in seconds")
    parser.add_argument("--mode", choices=["paper", "live"])
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.mode:
        config["trading"]["mode"] = args.mode

    setup_logging(config["logging"]["level"])

    trader = GemmaLocalTrader(
        config=config,
        symbols=args.symbols,
        interval=args.interval,
    )

    if args.once:
        trader.run_cycle()
    else:
        trader.start(run_every_seconds=args.every)


if __name__ == "__main__":
    main()
