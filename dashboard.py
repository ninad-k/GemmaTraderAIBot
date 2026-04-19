"""
Rey Capital AI Bot — Dashboard
================================
Real-time web dashboard with Flask + SocketIO showing
Gemma's trading decisions, open trades, P&L, and AI learning.

Usage (standalone):
    python dashboard.py --port 8050

Normally started via run.py which integrates SocketIO.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, render_template, jsonify, send_from_directory, request

logger = logging.getLogger("dashboard")

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
CONFIG_PATH = BASE_DIR / "config.yaml"
DECISIONS_LOG = LOGS_DIR / "gemma_decisions.json"
TRADES_LOG = LOGS_DIR / "trades.json"
OUTCOMES_LOG = LOGS_DIR / "trade_outcomes.json"
ADAPTIVE_CTX = LOGS_DIR / "adaptive_context.txt"
PARAM_ADJ_LOG = LOGS_DIR / "parameter_adjustments.json"


def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def read_json_log(path: Path) -> list:
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8-sig").strip()
            if text:
                return json.loads(text)
    except Exception as e:
        logger.warning(f"Error reading {path}: {e}")
    return []


# ─── Pages ───

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        str(BASE_DIR / "static"), "ReyCapital_Icon.png",
        mimetype="image/png"
    )


# ─── API Endpoints ───

@app.route("/api/health")
def api_health():
    config = load_config()
    trades = read_json_log(TRADES_LOG)
    decisions = read_json_log(DECISIONS_LOG)

    # Try to get real balance from MT5
    balance = 100000  # fallback
    try:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info:
            balance = info.balance
    except Exception:
        pass

    return jsonify({
        "status": "running",
        "mode": config.get("trading", {}).get("mode", "paper"),
        "model": config.get("ollama", {}).get("model", "gemma4"),
        "balance": balance,
        "open_trades": len([t for t in trades if not t.get("closed")]),
        "daily_pnl": 0.0,
        "total_decisions": len(decisions),
        "total_trades": len(trades),
        "symbols": config.get("trading", {}).get("allowed_symbols", []),
        "confidence_threshold": config.get("trading", {}).get("confidence_threshold", 0.60),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/decisions")
def api_decisions():
    decisions = read_json_log(DECISIONS_LOG)
    return jsonify({
        "decisions": decisions,
        "total": len(decisions),
    })


@app.route("/api/trades")
def api_trades():
    trades = read_json_log(TRADES_LOG)
    return jsonify({
        "trades": trades,
        "total": len(trades),
    })


@app.route("/api/stats")
def api_stats():
    decisions = read_json_log(DECISIONS_LOG)
    trades = read_json_log(TRADES_LOG)

    # Decision breakdown
    actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    confidences = []
    for d in decisions:
        dec = d.get("decision", {})
        action = dec.get("action", "HOLD")
        actions[action] = actions.get(action, 0) + 1
        confidences.append(dec.get("confidence", 0))

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Per-symbol stats
    symbol_stats = {}
    for d in decisions:
        sym = d.get("symbol", "?")
        dec = d.get("decision", {})
        if sym not in symbol_stats:
            symbol_stats[sym] = {
                "total": 0, "buys": 0, "sells": 0, "holds": 0, "avg_conf": [],
            }
        symbol_stats[sym]["total"] += 1
        action = dec.get("action", "HOLD")
        if action == "BUY":
            symbol_stats[sym]["buys"] += 1
        elif action == "SELL":
            symbol_stats[sym]["sells"] += 1
        else:
            symbol_stats[sym]["holds"] += 1
        symbol_stats[sym]["avg_conf"].append(dec.get("confidence", 0))

    for sym in symbol_stats:
        confs = symbol_stats[sym]["avg_conf"]
        symbol_stats[sym]["avg_conf"] = (
            round(sum(confs) / len(confs), 3) if confs else 0
        )

    return jsonify({
        "total_decisions": len(decisions),
        "total_trades": len(trades),
        "action_breakdown": actions,
        "avg_confidence": round(avg_confidence, 3),
        "symbol_stats": symbol_stats,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/symbols")
def api_symbols():
    """Get latest data for each symbol."""
    decisions = read_json_log(DECISIONS_LOG)
    latest = {}
    for d in decisions:
        sym = d.get("symbol")
        if sym:
            latest[sym] = d
    return jsonify({"symbols": latest})


@app.route("/api/learning")
def api_learning():
    """Get AI adaptive learning data: context, outcomes, adjustments."""
    # Adaptive context
    adaptive_context = ""
    try:
        if ADAPTIVE_CTX.exists():
            adaptive_context = ADAPTIVE_CTX.read_text().strip()
    except Exception:
        pass

    # Trade outcomes
    outcomes = read_json_log(OUTCOMES_LOG)

    # Win rate calculation
    wins = sum(1 for o in outcomes if o.get("profit", 0) > 0)
    losses = sum(1 for o in outcomes if o.get("profit", 0) <= 0)
    win_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0

    # Parameter adjustments
    adjustments = read_json_log(PARAM_ADJ_LOG)

    # Per-symbol performance from outcomes
    symbol_perf = {}
    for o in outcomes:
        sym = o.get("symbol", "?")
        if sym not in symbol_perf:
            symbol_perf[sym] = {"wins": 0, "losses": 0, "total_pnl": 0}
        if o.get("profit", 0) > 0:
            symbol_perf[sym]["wins"] += 1
        else:
            symbol_perf[sym]["losses"] += 1
        symbol_perf[sym]["total_pnl"] += o.get("profit", 0)

    return jsonify({
        "adaptive_context": adaptive_context,
        "total_outcomes": len(outcomes),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "symbol_performance": symbol_perf,
        "adjustments": adjustments[-10:],  # last 10 adjustments
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/pnl")
def api_pnl():
    """Get real-time P&L from MT5 open positions."""
    positions = []
    total_pnl = 0.0

    try:
        import MetaTrader5 as mt5
        mt5_positions = mt5.positions_get()
        if mt5_positions:
            for pos in mt5_positions:
                if pos.magic == 240411:  # Only our bot's trades
                    positions.append({
                        "symbol": pos.symbol,
                        "type": "BUY" if pos.type == 0 else "SELL",
                        "volume": pos.volume,
                        "price_open": pos.price_open,
                        "price_current": pos.price_current,
                        "profit": pos.profit,
                        "swap": pos.swap,
                        "sl": pos.sl,
                        "tp": pos.tp,
                    })
                    total_pnl += pos.profit + pos.swap
    except Exception:
        pass

    return jsonify({
        "positions": positions,
        "total_unrealized_pnl": round(total_pnl, 2),
        "count": len(positions),
        "timestamp": datetime.now().isoformat(),
    })


JOURNAL_LOG = LOGS_DIR / "trade_journal.json"


@app.route("/api/journal")
def api_journal():
    """Get trade journal with full Gemma thinking and user comments."""
    journal = read_json_log(JOURNAL_LOG)
    return jsonify({
        "entries": journal,
        "total": len(journal),
    })


@app.route("/api/journal/comment", methods=["POST"])
def api_journal_comment():
    """Add a user comment to a journal entry."""
    data = request.get_json()
    if not data or "trade_id" not in data or "comment" not in data:
        return jsonify({"error": "Missing trade_id or comment"}), 400

    journal = read_json_log(JOURNAL_LOG)
    for entry in journal:
        if entry.get("trade_id") == data["trade_id"]:
            if "comments" not in entry:
                entry["comments"] = []
            entry["comments"].append({
                "text": data["comment"],
                "timestamp": datetime.now().isoformat(),
            })
            break

    try:
        JOURNAL_LOG.write_text(json.dumps(journal, indent=2))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok"})


@app.route("/api/gemma_thinking")
def api_gemma_thinking():
    """Get recent Gemma thinking/reasoning debug logs."""
    decisions = read_json_log(DECISIONS_LOG)
    # Return last 20 with raw thinking
    recent = decisions[-20:] if len(decisions) > 20 else decisions
    return jsonify({
        "decisions": recent,
        "total": len(decisions),
    })


@app.route("/api/lot_override", methods=["POST"])
def api_lot_override():
    """Set manual lot size override per symbol."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    override_path = LOGS_DIR / "lot_overrides.json"
    overrides = {}
    try:
        if override_path.exists():
            text = override_path.read_text(encoding="utf-8-sig").strip()
            if text:
                overrides = json.loads(text)
    except Exception:
        pass

    if "symbol" in data and "lot_size" in data:
        if data["lot_size"] is None or data["lot_size"] == 0:
            overrides.pop(data["symbol"], None)  # Remove override
        else:
            overrides[data["symbol"]] = float(data["lot_size"])

    override_path.write_text(json.dumps(overrides, indent=2))
    return jsonify({"status": "ok", "overrides": overrides})


@app.route("/api/lot_overrides")
def api_lot_overrides():
    """Get current lot size overrides."""
    override_path = LOGS_DIR / "lot_overrides.json"
    overrides = {}
    try:
        if override_path.exists():
            text = override_path.read_text(encoding="utf-8-sig").strip()
            if text:
                overrides = json.loads(text)
    except Exception:
        pass
    return jsonify({"overrides": overrides})


# ─── Main (standalone mode) ───

def main():
    parser = argparse.ArgumentParser(description="Rey Capital AI Bot Dashboard")
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("PORT", 8050)),
        help="Dashboard port",
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(f"""
+=====================================================+
|       REY CAPITAL AI BOT — DASHBOARD                 |
+=====================================================+
|  URL:  http://localhost:{args.port:<37}|
|  Logs: {str(LOGS_DIR):<46}|
+=====================================================+
    """)

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
