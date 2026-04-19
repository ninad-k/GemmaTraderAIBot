"""
Rey Capital AI Bot — Unified Entry Point
==========================================
Single entry point that starts:
  1. Flask + SocketIO dashboard (web UI)
  2. GemmaLocalTrader in background thread (trading engine)
  3. TradeReviewer for self-learning

Usage:
    python run.py                         # defaults from config.yaml
    python run.py --port 8050             # custom dashboard port
    python run.py --mode paper            # paper trading
    python run.py --symbols US100_Spot    # specific symbols only

Dashboard: http://localhost:8050
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

import yaml
from flask_socketio import SocketIO

from dashboard import app, LOGS_DIR
from local_trader import GemmaLocalTrader, setup_logging
from trade_reviewer import TradeReviewer

logger = logging.getLogger("rey_capital")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def main():
    parser = argparse.ArgumentParser(description="Rey Capital AI Bot")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--port", type=int, default=None, help="Dashboard port")
    parser.add_argument("--symbols", nargs="+", help="Override symbols")
    parser.add_argument("--interval", default=None, help="Candle interval")
    parser.add_argument("--mode", choices=["paper", "live"])
    parser.add_argument("--no-trade", action="store_true",
                        help="Dashboard only, no trading")
    args = parser.parse_args()

    # Load config
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.mode:
        config["trading"]["mode"] = args.mode

    port = args.port or int(os.environ.get("PORT", config.get("server", {}).get("port", 8050)))

    # Setup logging
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    LOGS_DIR.mkdir(exist_ok=True)

    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    logger.info(f"""
+===========================================================+
|               REY CAPITAL AI BOT                           |
+===========================================================+
|  Dashboard:  http://localhost:{port:<33}|
|  Mode:       {config['trading']['mode'].upper():<44}|
|  Model:      {config['ollama']['model']:<44}|
|  Symbols:    {', '.join(args.symbols or config['trading']['allowed_symbols']):<44}|
|  Interval:   {(args.interval or config.get('mt5_data',{}).get('timeframe','1m')):<44}|
|  Trading:    {'DISABLED' if args.no_trade else 'ENABLED':<44}|
+===========================================================+
    """)

    if not args.no_trade:
        # Initialize trader
        trader = GemmaLocalTrader(
            config=config,
            symbols=args.symbols,
            interval=args.interval,
        )
        trader.socketio = socketio

        # Initialize trade reviewer
        reviewer = TradeReviewer(config, risk_manager=trader.risk_manager)

        # Start trader in background thread
        def trader_loop():
            """Background trading loop that integrates with the reviewer."""
            time.sleep(3)  # Let Flask start first
            logger.info("Trading engine started")

            poll_interval = config.get("mt5_data", {}).get(
                "poll_interval_seconds", 60
            )

            # Run first cycle
            try:
                trader.run_cycle()
            except Exception as e:
                logger.error(f"First cycle error: {e}", exc_info=True)

            while True:
                try:
                    time.sleep(poll_interval)
                    trader.run_cycle()

                    # Run performance review periodically
                    summary = reviewer.analyze_performance()
                    if summary:
                        socketio.emit("stats_update", {
                            "win_rate": summary.get("win_rate"),
                            "total_pnl": summary.get("total_pnl"),
                            "timestamp": summary.get("timestamp"),
                        })

                    # Weekly review check (Gemma reviews itself)
                    reviewer.weekly_review()

                except Exception as e:
                    logger.error(f"Cycle error: {e}", exc_info=True)
                    time.sleep(10)

        trade_thread = threading.Thread(target=trader_loop, daemon=True)
        trade_thread.start()

    # SocketIO event handlers
    @socketio.on("connect")
    def on_connect():
        logger.debug("Client connected to WebSocket")

    @socketio.on("disconnect")
    def on_disconnect():
        logger.debug("Client disconnected from WebSocket")

    # Start Flask + SocketIO
    socketio.run(
        app,
        host=config.get("server", {}).get("host", "0.0.0.0"),
        port=port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
