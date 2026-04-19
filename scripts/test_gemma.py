"""
Quick test script — sends a fake TradingView alert to verify the pipeline works.

Usage:
    1. Start the server:  python server.py
    2. In another terminal: python test_gemma.py
"""

import requests
import json

WEBHOOK_URL = "http://localhost:5000/webhook"

# Simulated TradingView alert data
fake_alert = {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE",
    "timeframe": "60",
    "close": 68450.50,
    "open": 67800.00,
    "high": 68700.00,
    "low": 67600.00,
    "volume": 125000,
    "rsi": 42.5,
    "macd": 125.4,
    "macd_signal": 98.2,
    "macd_hist": 27.2,
    "ema_fast": 68100.00,
    "ema_slow": 67500.00,
    "atr": 850.00,
    "bb_upper": 69200.00,
    "bb_lower": 66800.00,
    "bb_pos": "INSIDE",
    "vwap": 68050.00,
    "trend": "BULLISH",
    "ema_cross": "NONE",
    "vol_trend": "ABOVE_AVG"
}

print("Sending fake alert to webhook...")
print(f"Symbol: {fake_alert['symbol']} | Close: {fake_alert['close']}")
print("-" * 50)

try:
    response = requests.post(WEBHOOK_URL, json=fake_alert, timeout=120)
    result = response.json()
    print(json.dumps(result, indent=2))
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect. Is the server running? (python server.py)")
except Exception as e:
    print(f"❌ Error: {e}")
