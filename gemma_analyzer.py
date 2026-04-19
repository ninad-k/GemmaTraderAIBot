"""
Rey Capital AI Bot — Gemma 4 Analyzer
=======================================
Sends comprehensive market data (30+ indicators) to Gemma 4
via Ollama and returns trade decisions.

Self-improving: loads adaptive context from past trade outcomes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are Rey Capital's AI crypto scalping bot. You analyze 1-minute candle data across 30+ technical indicators and make precise, profitable trade decisions on cryptocurrency CFDs.

INSTRUMENTS (all 24/7 crypto CFDs):
- BTCUSD: Bitcoin — king of crypto, highest liquidity, drives market sentiment. $70K+ range. Moves in waves, respect momentum.
- ETHUSD: Ethereum — second largest, correlates with BTC but has its own DeFi/network catalysts. More volatile intraday.
- LTCUSD: Litecoin — fast settlement, often leads BTC moves. Lower liquidity = sharper spikes.
- XRPUSD: Ripple — news-driven, regulatory sensitive. Can spike 2-5% in minutes on headlines.
- SOLUSD: Solana — high-beta altcoin, amplifies BTC moves 2-3x. Best for momentum trades.

CRYPTO-SPECIFIC RULES:
1. Crypto trades 24/7 — there is ALWAYS opportunity. Do NOT default to HOLD.
2. Crypto is MOMENTUM-driven: when momentum aligns, ride it. When it fades, exit fast.
3. BTC leads — if BTC is strongly trending, altcoins will follow with amplified moves.
4. Volume is CRITICAL in crypto — a move without volume is a fake move. Always check vol_trend.
5. Crypto respects Ichimoku Cloud and EMA structure well on 1M timeframe.
6. RSI extremes matter more in crypto: oversold (<25) = bounce likely, overbought (>75) = pullback likely.

YOUR APPROACH:
1. You are a 1-minute crypto scalper. Prioritize MOMENTUM and PRICE ACTION.
2. Be AGGRESSIVE — take BUY and SELL trades when 2+ signals align. Crypto rewards decisiveness.
3. Look for confluence: at least 2-3 indicators agreeing (e.g., RSI + MACD + Ichimoku + candlestick pattern).
4. Key signals for crypto:
   - Ichimoku Cloud: price vs cloud, TK cross, cloud color — very reliable on crypto 1M
   - MACD histogram direction: rising = buy momentum, falling = sell momentum
   - Volume surges: confirm breakouts ONLY with above-average volume
   - Candlestick patterns at S/R levels: engulfing, hammer, shooting star
   - Supertrend + EMA alignment: when both agree, high-probability trade
5. Use ATR for SL/TP — crypto ATR is wider, keep risk tight (SL: 0.5-1.5 ATR, TP: 1.0-2.0 ATR).
6. You learn from your mistakes. Adaptive context from past trades is provided — follow its lessons strictly.

PROFIT FOCUS:
- Only take trades where reward > risk (TP > SL distance)
- Cut losses fast (tight SL), let winners run (wider TP)
- If a trade setup is marginal, SKIP IT. Wait for the next clear signal.
- Track your win rate mentally — if losing, be more selective. If winning, maintain discipline.

RULES:
- Respond with ONLY valid JSON — no explanation, no markdown, no code blocks.
- confidence must be a decimal number between 0.0 and 1.0 (NOT a string like "HIGH").
- When signals clearly align: BUY or SELL with confidence 0.7-0.95.
- When signals partially align: BUY or SELL with confidence 0.5-0.7.
- Only HOLD when indicators strongly conflict or volume is dead.

JSON Response Format:
{"action": "BUY|SELL|HOLD", "confidence": 0.85, "sl_distance_atr": 1.0, "tp_distance_atr": 1.5, "reason": "concise 1-line reason"}
"""


def analyze_with_gemma(market_data: dict, config: dict) -> dict:
    """
    Send market data to Gemma 4 and get a trade decision.
    Includes adaptive context from past trade outcomes.
    """
    prompt = _build_prompt(market_data)

    # Load adaptive context (self-learning from past trades)
    adaptive_context = _load_adaptive_context(config)
    system = SYSTEM_PROMPT
    if adaptive_context:
        system = system + "\n\n" + adaptive_context

    try:
        response = requests.post(
            config["ollama"]["url"],
            json={
                "model": config["ollama"]["model"],
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config["ollama"].get("temperature", 0.1) or 0.1,
                    "num_predict": config["ollama"].get("num_predict", 8192),
                },
            },
            timeout=config["ollama"]["timeout"],
        )
        response.raise_for_status()

        raw_response = response.json()["response"].strip()
        logger.info(f"Gemma raw response for {market_data.get('symbol', '?')}:\n{raw_response}")

        decision = _parse_response(raw_response)
        decision["timestamp"] = datetime.now().isoformat()
        decision["symbol"] = market_data.get("symbol", "UNKNOWN")
        decision = _validate_decision(decision)

        # Include Gemma's full thinking for journal/debug
        decision["raw_gemma_response"] = raw_response
        decision["prompt_sent"] = prompt
        decision["indicators_summary"] = {
            "rsi": market_data.get("rsi"),
            "macd_hist": market_data.get("macd_hist"),
            "trend": market_data.get("trend"),
            "ichimoku_signal": market_data.get("ichimoku_signal"),
            "supertrend": market_data.get("supertrend"),
            "adx": market_data.get("adx"),
            "vol_trend": market_data.get("vol_trend"),
            "vol_ratio": market_data.get("vol_ratio"),
            "bb_pos": market_data.get("bb_pos"),
            "ema_cross": market_data.get("ema_cross"),
            "candle_patterns": market_data.get("candle_patterns"),
            "stoch_rsi_k": market_data.get("stoch_rsi_k"),
            "cci": market_data.get("cci"),
            "williams_r": market_data.get("williams_r"),
            "mfi": market_data.get("mfi"),
            "psar_signal": market_data.get("psar_signal"),
        }

        return decision

    except requests.exceptions.Timeout:
        logger.error("Gemma 4 request timed out")
        return _hold_decision("timeout")
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama — is it running?")
        return _hold_decision("connection_error")
    except Exception as e:
        logger.error(f"Gemma analysis failed: {e}")
        return _hold_decision(f"error: {str(e)}")


def _build_prompt(data: dict) -> str:
    """Build a comprehensive prompt from 30+ indicators."""
    symbol = data.get("symbol", "UNKNOWN")
    tf = data.get("timeframe", "1m")

    return f"""Analyze {symbol} on {tf} timeframe. Make a trade decision.

PRICE ACTION:
  Close: {data.get('close')} | Open: {data.get('open')} | High: {data.get('high')} | Low: {data.get('low')}
  Last 5 Candles: {data.get('last_5_candles', 'N/A')}
  Candlestick Patterns: {data.get('candle_patterns', 'NONE')}
  Nearest Support: {data.get('nearest_support', 'N/A')} | Resistance: {data.get('nearest_resistance', 'N/A')}

TREND:
  EMA(9): {data.get('ema9')} | EMA(20): {data.get('ema20')} | EMA(50): {data.get('ema50')} | EMA(200): {data.get('ema200')}
  SMA(20): {data.get('sma20')} | SMA(50): {data.get('sma50')}
  Trend: {data.get('trend')} | EMA Cross: {data.get('ema_cross', 'NONE')}
  ADX: {data.get('adx')} | DI+: {data.get('di_plus')} | DI-: {data.get('di_minus')}
  Supertrend: {data.get('supertrend', 'N/A')}
  Parabolic SAR: {data.get('psar_signal', 'N/A')}

ICHIMOKU CLOUD:
  Tenkan-sen: {data.get('ichimoku_tenkan')} | Kijun-sen: {data.get('ichimoku_kijun')}
  Span A: {data.get('ichimoku_span_a')} | Span B: {data.get('ichimoku_span_b')}
  Signal: {data.get('ichimoku_signal', 'N/A')} | Cloud: {data.get('ichimoku_cloud_color', 'N/A')}

MOMENTUM:
  RSI(14): {data.get('rsi')}
  MACD: {data.get('macd')} | Signal: {data.get('macd_signal')} | Hist: {data.get('macd_hist')}
  Stoch RSI K: {data.get('stoch_rsi_k')} | D: {data.get('stoch_rsi_d')}
  Stoch K: {data.get('stoch_k')} | D: {data.get('stoch_d')}
  CCI(20): {data.get('cci')} | Williams %R(14): {data.get('williams_r')}
  ROC(10): {data.get('roc')} | MFI(14): {data.get('mfi')}

VOLATILITY:
  ATR(14): {data.get('atr')}
  BB Upper: {data.get('bb_upper')} | Mid: {data.get('bb_mid')} | Lower: {data.get('bb_lower')}
  BB Position: {data.get('bb_pos')} | BB Width: {data.get('bb_width')}%

VOLUME:
  Current Volume: {data.get('volume')} | Trend: {data.get('vol_trend')} ({data.get('vol_ratio', '?')}x avg)
  VWAP: {data.get('vwap')}

Respond with JSON only."""


def _load_adaptive_context(config: dict) -> str:
    """Load adaptive context from past trade analysis (self-learning)."""
    try:
        ctx_path = Path(config.get("logging", {}).get(
            "adaptive_context", "logs/adaptive_context.txt"
        ))
        if ctx_path.exists():
            text = ctx_path.read_text().strip()
            if text:
                return f"ADAPTIVE CONTEXT (lessons from your past trades):\n{text}"
    except Exception:
        pass
    return ""


def _parse_response(raw: str) -> dict:
    """Parse JSON from Gemma's response, handling code blocks."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
        raise ValueError(f"Could not parse JSON from: {raw}")


def _validate_decision(decision: dict) -> dict:
    """Validate and sanitize the decision."""
    valid_actions = {"BUY", "SELL", "HOLD"}

    if decision.get("action", "").upper() not in valid_actions:
        decision["action"] = "HOLD"
        decision["confidence"] = 0.0
        decision["reason"] = "invalid action from model"
    else:
        decision["action"] = decision["action"].upper()

    raw_conf = decision.get("confidence", 0)
    # Handle string confidence values from Gemma (e.g., "LOW", "HIGH")
    if isinstance(raw_conf, str):
        conf_map = {
            "very low": 0.15, "low": 0.3, "medium": 0.5, "moderate": 0.55,
            "high": 0.75, "very high": 0.9,
        }
        conf = conf_map.get(raw_conf.strip().lower(), 0.5)
    else:
        conf = float(raw_conf)
    decision["confidence"] = max(0.0, min(1.0, conf))

    decision.setdefault("sl_distance_atr", 1.0)
    decision.setdefault("tp_distance_atr", 1.5)
    decision.setdefault("reason", "no reason given")

    # Clamp SL/TP to reasonable 1M scalping ranges
    sl = float(decision["sl_distance_atr"])
    tp = float(decision["tp_distance_atr"])
    decision["sl_distance_atr"] = max(0.5, min(2.0, sl))
    decision["tp_distance_atr"] = max(0.75, min(3.0, tp))

    return decision


def _hold_decision(reason: str) -> dict:
    """Return a safe HOLD decision."""
    return {
        "action": "HOLD",
        "confidence": 0.0,
        "sl_distance_atr": 0,
        "tp_distance_atr": 0,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "symbol": "UNKNOWN",
    }


def review_trades_with_gemma(trade_history: list, config: dict) -> str:
    """
    Feed trade history back to Gemma for self-review.
    Returns updated trading rules/lessons learned.
    Used by trade_reviewer.py for weekly meta-analysis.
    """
    if not trade_history:
        return ""

    trades_text = json.dumps(trade_history[-50:], indent=2)

    prompt = f"""Review these recent trades and their outcomes.
Analyze what patterns worked and what didn't.
Generate concise trading lessons (max 10 bullet points) that should guide future decisions.

Focus on:
1. Which indicator combinations led to wins vs losses
2. Which symbols performed best/worst
3. What market conditions to avoid
4. What entry patterns to prioritize

Trades:
{trades_text}

Respond with a plain text list of lessons (no JSON, no markdown)."""

    try:
        response = requests.post(
            config["ollama"]["url"],
            json={
                "model": config["ollama"]["model"],
                "system": "You are a trading performance analyst. Review trades and extract actionable lessons.",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 500},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        logger.error(f"Trade review failed: {e}")
        return ""
