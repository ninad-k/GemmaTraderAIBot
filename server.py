"""
Gemma 4 Auto Trader — Webhook Server
=====================================
Receives TradingView alerts via webhook, analyzes with Gemma 4,
and executes trades through the configured broker.

Usage:
    python server.py                    # start with default config
    python server.py --config my.yaml   # start with custom config
    python server.py --mode paper       # force paper trading mode

Setup:
    1. Add the PineScript indicator to your TradingView chart
    2. Create an alert → Webhook URL: http://<your-ip>:5000/webhook
    3. Start this server
    4. Gemma 4 analyzes each alert and decides BUY/SELL/HOLD
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, request, jsonify

from gemma_analyzer import analyze_with_gemma
from risk_manager import RiskManager
from broker_bridge import create_broker

# ─── Setup ───
app = Flask(__name__)
logger = logging.getLogger("gemma_trader")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Globals (initialized in main) ───
config = None
risk_manager = None
broker = None
decision_log = []


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Receive TradingView alert webhook.
    Expected JSON payload from PineScript alert.
    """
    try:
        # Parse incoming data
        if request.content_type == "application/json":
            data = request.json
        else:
            # TradingView sometimes sends as text
            data = json.loads(request.data.decode("utf-8"))

        logger.info(f"📨 Alert received: {data.get('symbol', '?')} @ {data.get('close', '?')}")

        # 1. Analyze with Gemma 4
        decision = analyze_with_gemma(data, config)
        logger.info(f"🤖 Gemma decision: {decision['action']} "
                     f"(confidence: {decision['confidence']:.2f}) — {decision['reason']}")

        # Log decision
        _log_decision(data, decision)

        # 2. Check if action is HOLD
        if decision["action"] == "HOLD":
            return jsonify({
                "status": "hold",
                "decision": decision
            }), 200

        # 3. Risk check
        allowed, reason = risk_manager.can_trade(decision, data)
        if not allowed:
            logger.warning(f"⛔ Trade blocked: {reason}")
            return jsonify({
                "status": "blocked",
                "reason": reason,
                "decision": decision
            }), 200

        # 4. Calculate position size
        balance = broker.get_balance()
        atr = float(data.get("atr", 0))
        pos_size = risk_manager.calculate_position_size(
            balance, atr, decision.get("sl_distance_atr", 1.5)
        )

        if pos_size["qty"] <= 0:
            return jsonify({"status": "error", "reason": "invalid position size"}), 200

        # 5. Calculate SL/TP prices
        close = float(data["close"])
        sl_distance = atr * decision.get("sl_distance_atr", 1.5)
        tp_distance = atr * decision.get("tp_distance_atr", 2.5)

        if decision["action"] == "BUY":
            sl_price = round(close - sl_distance, 2)
            tp_price = round(close + tp_distance, 2)
        else:  # SELL
            sl_price = round(close + sl_distance, 2)
            tp_price = round(close - tp_distance, 2)

        # 6. Execute trade
        result = broker.place_order(
            symbol=data["symbol"],
            action=decision["action"],
            qty=pos_size["qty"],
            sl=sl_price,
            tp=tp_price
        )

        # 7. Register with risk manager
        if result.get("status") == "filled":
            risk_manager.register_trade({
                "symbol": data["symbol"],
                "action": decision["action"],
                "qty": pos_size["qty"],
                "entry_price": close,
                "sl": sl_price,
                "tp": tp_price,
                "confidence": decision["confidence"],
                "reason": decision["reason"],
                "timestamp": datetime.now().isoformat()
            })

        return jsonify({
            "status": "executed" if result.get("status") == "filled" else "failed",
            "decision": decision,
            "position_size": pos_size,
            "sl": sl_price,
            "tp": tp_price,
            "broker_result": result
        }), 200

    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "model": config["ollama"]["model"],
        "mode": config["trading"]["mode"],
        "open_trades": len(risk_manager.open_trades),
        "daily_pnl": risk_manager.daily_pnl,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/trades", methods=["GET"])
def get_trades():
    """Get open trades."""
    return jsonify({
        "open_trades": risk_manager.open_trades,
        "total": len(risk_manager.open_trades)
    })


@app.route("/decisions", methods=["GET"])
def get_decisions():
    """Get recent Gemma decisions."""
    return jsonify({
        "decisions": decision_log[-50:],  # last 50
        "total": len(decision_log)
    })


@app.route("/close/<symbol>", methods=["POST"])
def close_trade(symbol: str):
    """Manually close a trade."""
    result = broker.close_position(symbol)
    if result.get("status") == "closed":
        risk_manager.close_trade(symbol, 0)  # PnL calculated externally
    return jsonify(result)


def _log_decision(market_data: dict, decision: dict):
    """Log Gemma's decision for analysis."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": market_data.get("symbol"),
        "close": market_data.get("close"),
        "decision": decision
    }
    decision_log.append(entry)

    # Also write to file
    try:
        log_path = Path(config["logging"]["decision_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logs = []
        if log_path.exists():
            logs = json.loads(log_path.read_text())
        logs.append(entry)
        # Keep last 1000 decisions
        log_path.write_text(json.dumps(logs[-1000:], indent=2))
    except Exception as e:
        logger.warning(f"Could not write decision log: {e}")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/server.log")
        ]
    )


def main():
    global config, risk_manager, broker

    parser = argparse.ArgumentParser(description="Gemma 4 Auto Trader")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--mode", choices=["paper", "live"], help="Override trading mode")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.mode:
        config["trading"]["mode"] = args.mode

    # Setup
    Path("logs").mkdir(exist_ok=True)
    setup_logging(config["logging"]["level"])

    logger.info("=" * 60)
    logger.info("  GEMMA 4 AUTO TRADER")
    logger.info(f"  Mode:  {config['trading']['mode'].upper()}")
    logger.info(f"  Model: {config['ollama']['model']}")
    logger.info(f"  Port:  {config['server']['port']}")
    logger.info("=" * 60)

    # Initialize components
    risk_manager = RiskManager(config)
    broker = create_broker(config)

    logger.info(f"Broker balance: {broker.get_balance()}")
    logger.info(f"Webhook URL: http://0.0.0.0:{config['server']['port']}/webhook")
    logger.info("Waiting for TradingView alerts...")

    # Start server
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=False
    )


if __name__ == "__main__":
    main()
