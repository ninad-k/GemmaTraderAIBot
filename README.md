# GemmaTraderAIBot

A self-learning AI crypto scalping bot driven by Google's **Gemma** LLM (run locally via [Ollama](https://ollama.com)) and executed through **MetaTrader 5**. Includes a Flask + SocketIO live dashboard and a statistically-rigorous self-review loop that adapts the model's prompt based on validated patterns from its own trade history.

## Features

- **LLM-driven decisions** — Gemma analyzes 30+ technical indicators per 1-minute candle and returns structured JSON trade decisions.
- **Self-learning with guardrails** — the adaptive loop uses walk-forward validation, binomial significance testing, regime scoping, and freshness decay so that small-sample noise is not mistaken for edge.
- **Risk layer** — per-symbol cooldowns, loss-streak cooldowns, daily loss cap, dynamic confidence threshold, ATR-based SL/TP, MT5-aware lot sizing.
- **Live dashboard** — Flask + SocketIO UI for monitoring trades, PnL, and Gemma's reasoning in real time.
- **Paper / live modes** — swap via config or CLI flag.

## Architecture

```
run.py                   Unified entry point (dashboard + trader + reviewer)
  |
  +-- local_trader.py    GemmaLocalTrader - main polling loop
  +-- gemma_analyzer.py  Sends market data to Ollama, parses decisions
  +-- risk_manager.py    Validates trades, tracks outcomes, adjusts threshold
  +-- trade_reviewer.py  Walk-forward validated pattern lessons
  +-- broker_bridge.py   MT5 order execution
  +-- mt5_data_feed.py   MT5 market data
  +-- dashboard.py       Flask + SocketIO web UI
  +-- server.py          Webhook / API endpoints
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with a Gemma model pulled (`ollama pull gemma3:4b` or larger)
- MetaTrader 5 terminal installed and logged into your broker
- Windows (MT5 Python bindings are Windows-only)

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.yaml`:

- `ollama.model` — the local Gemma tag (e.g. `gemma3:4b`, `gemma3:12b`)
- `broker.mt5.login / password / server` — your MT5 credentials (or leave blank and log in via MT5 terminal)
- `trading.mode` — `paper` or `live`
- `trading.allowed_symbols` — the instruments you allow

Start Ollama and pull the model:

```bash
ollama pull gemma3:4b
ollama serve
```

Run:

```bash
python run.py
```

Dashboard: http://localhost:8050

### CLI flags

```
python run.py --mode paper
python run.py --symbols BTCUSD ETHUSD
python run.py --port 8080
python run.py --no-trade          # dashboard only
```

## Self-learning loop

Every `adaptive.review_every_n_trades` trades, `trade_reviewer.py` regenerates an adaptive context block that is prepended to Gemma's system prompt. A pattern (e.g. "RSI oversold under HIGH_VOL regime on SOLUSD") is promoted into the live prompt only if:

1. It has >=30 samples on an older *train* slice.
2. It re-holds on a held-out *test* slice (>=10 samples).
3. It re-holds on the most recent window (>=15 samples).
4. Its combined sample clears a binomial significance test (p < 0.05 vs 50% baseline).
5. Its direction (favor / avoid) agrees across all three slices.

Patterns that stop holding on the recent window are dropped automatically — no lesson ossifies.

Tune all thresholds under `adaptive:` in `config.yaml`.

> The narrative "Gemma reviews itself" output is kept only as an advisory artifact (`logs/gemma_narrative_review.txt`) and is **not** auto-fed back into the live prompt. A 4B model cannot reliably critique its own trades.

## Logs

All logs are written under `logs/` (gitignored):

- `trades.json` — all opened trades
- `trade_outcomes.json` — closed trades with full indicator snapshots
- `gemma_decisions.json` — raw Gemma responses
- `adaptive_context.txt` — current validated-pattern advice (live)
- `gemma_narrative_review.txt` — advisory self-review (not live)
- `parameter_adjustments.json` — threshold change history

## Disclaimer

This is a research / educational project. Automated trading carries substantial risk of loss. Paper-trade first. Use real capital only after you have validated sustained performance across multiple market regimes. The authors accept no liability for trading losses.

## License

See `LICENSE`.
