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

import requests
import yaml
from flask_socketio import SocketIO

from gemma_trader.dashboard import app, LOGS_DIR, attach_trader
from gemma_trader.local_trader import GemmaLocalTrader, setup_logging
from gemma_trader.trade_reviewer import TradeReviewer
from gemma_trader.safety import get_safety
from scripts.init_config import initialize_configs

logger = logging.getLogger("rey_capital")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _preflight_check(config: dict) -> bool:
    """
    Validate critical dependencies before trading.
    Returns True if all checks pass (or are optional).
    Exits with error if critical check fails.
    """
    logger.info("\n+===================== PRE-FLIGHT CHECKS =====================+")

    # 1. Validate config structure
    try:
        mode = config["trading"]["mode"]
        ollama_url = config["ollama"]["url"]
        ollama_model = config["ollama"]["model"]
        broker = config["broker"]["name"]
        logger.info("✓ Config keys validated")
    except KeyError as e:
        logger.error(f"✗ Missing required config key: {e}")
        sys.exit(1)

    # 2. Check Ollama reachability
    try:
        response = requests.get(ollama_url.replace("/api/generate", ""), timeout=2)
        response.raise_for_status()
        logger.info(f"✓ Ollama reachable ({ollama_url})")
    except requests.exceptions.Timeout:
        logger.error(f"✗ Ollama timeout: {ollama_url} (is `ollama serve` running?)")
        sys.exit(1)
    except (requests.exceptions.ConnectionError, requests.exceptions.RequestException):
        logger.error(f"✗ Cannot reach Ollama: {ollama_url}")
        logger.error("  Start it with: `ollama serve`")
        sys.exit(1)

    # 3. Check Ollama model availability
    try:
        model_check = requests.post(
            ollama_url,
            json={"model": ollama_model, "prompt": "test", "stream": False},
            timeout=5,
        )
        if model_check.status_code == 404:
            logger.warning(f"⚠️ Model '{ollama_model}' not in Ollama")
            logger.warning(f"  Pull it with: `ollama pull {ollama_model}`")
        elif model_check.status_code == 200:
            logger.info(f"✓ Model '{ollama_model}' available")
        else:
            logger.warning(f"⚠️ Model check returned {model_check.status_code}")
    except requests.exceptions.Timeout:
        logger.warning(f"⚠️ Model check timed out (Ollama may be busy)")
    except Exception as e:
        logger.warning(f"⚠️ Model check failed: {e}")

    # 4. Check log directory is writable
    try:
        logs_dir = Path(config.get("logging", {}).get("trade_log", "logs/trades.json")).parent
        logs_dir.mkdir(parents=True, exist_ok=True)
        test_file = logs_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        logger.info("✓ Log directory writable")
    except Exception as e:
        logger.error(f"✗ Cannot write to logs directory: {e}")
        sys.exit(1)

    # 5. Check trading mode and warn if live
    if mode == "live":
        logger.warning("⚠️ WARNING: Trading mode is LIVE")
        logger.warning("   See ROADMAP.md Phase 0 before going live with real capital")
    elif mode == "paper":
        logger.info("✓ Paper mode (safe for testing)")
    else:
        logger.warning(f"⚠️ Unknown mode '{mode}' (should be 'paper' or 'live')")

    # 6. Check MT5 if in live mode
    if mode == "live":
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                logger.warning(f"⚠️ MT5 initialization failed: {mt5.last_error()}")
            else:
                info = mt5.account_info()
                if info:
                    logger.info(f"✓ MT5 connected (balance: {info.balance})")
                else:
                    logger.warning("⚠️ MT5 login failed - check credentials in config.yaml")
        except ImportError:
            logger.warning("⚠️ MetaTrader5 not installed (OK for paper mode)")
        except Exception as e:
            logger.warning(f"⚠️ MT5 check failed: {e}")

    # 7. Check example config files
    examples = {
        "notifications.yaml": "notifications.yaml.example",
        "news_blackouts.yaml": "news_blackouts.yaml.example",
    }
    for real_name, example_name in examples.items():
        real_path = PROJECT_ROOT / real_name
        if not real_path.exists():
            logger.info(f"ℹ️ {real_name} not found (optional)")
            logger.info(f"   Copy it: cp {example_name} {real_name}")

    logger.info("+============================================================+\n")
    return True


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

    # Auto-generate missing example configs
    initialize_configs(PROJECT_ROOT)

    # Load config
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.mode:
        config["trading"]["mode"] = args.mode

    port = args.port or int(os.environ.get("PORT", config.get("server", {}).get("port", 8050)))

    # Setup logging first
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    LOGS_DIR.mkdir(exist_ok=True)

    # Run pre-flight checks
    _preflight_check(config)

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
        attach_trader(trader)

        # Watchdog: if no heartbeat for 3×poll_interval, log + notify
        def watchdog_loop():
            safety = get_safety(config)
            poll = config.get("mt5_data", {}).get("poll_interval_seconds", 60)
            threshold = max(poll * 3, 180)
            while True:
                time.sleep(poll)
                stale = safety.seconds_since_heartbeat()
                if stale and stale > threshold:
                    logger.error(f"[WATCHDOG] trader stalled for {stale:.0f}s")
                    try:
                        trader.notifier.notify(
                            "halt",
                            f"Watchdog: trader stalled for {stale:.0f}s",
                            {"threshold": threshold},
                        )
                    except Exception:
                        pass
        threading.Thread(target=watchdog_loop, daemon=True).start()

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



