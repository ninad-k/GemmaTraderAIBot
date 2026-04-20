# GemmaTraderAIBot

A self-learning crypto/CFD scalping bot driven by Google's **Gemma** LLM
(run locally via [Ollama](https://ollama.com)) and executed through
**MetaTrader 5** or **Binance**. Includes a Flask + SocketIO dashboard,
a settings UI, multi-channel alerts (Telegram / Microsoft Teams / WhatsApp),
a walk-forward-validated self-review loop, a kill-switch + drawdown
circuit breaker, and a backtester.

> **Read before trading live:** [ROADMAP.md](ROADMAP.md) — the phased plan
> for taking this from prototype to risk-adjusted profitable.

---

## Features

- **LLM-driven decisions** — Gemma analyses 30+ technical indicators per
  1-minute candle and returns structured JSON trade decisions.
- **Ensemble gate** — optionally requires 4B + 12B agreement before trading.
- **Feature-dedupe cache** — skips redundant LLM calls when candles barely
  move, cutting cost and latency.
- **Self-learning with guardrails** — walk-forward validation, binomial
  significance testing, regime scoping, and freshness decay so that
  small-sample noise is not mistaken for edge.
- **Risk & safety layer** — per-symbol cooldowns, loss-streak cooldowns,
  daily loss cap, dynamic confidence threshold, ATR-based SL/TP,
  MT5-aware lot sizing, kill-switch, drawdown circuit breaker, watchdog
  for stalled loop, MT5 reconnect.
- **Broker symbol aliasing** — generic names (`GOLD`) map to broker tickers
  (`XAUUSD` on IC Markets, `XAUUSD_` on CFI) via a hot-reloaded registry.
- **Multi-channel alerts** — Telegram + Microsoft Teams + WhatsApp (Meta
  Cloud API), with per-event toggles.
- **News blackout windows** — skip new entries during FOMC / CPI / etc.
- **Metrics + backtest** — Sharpe, Sortino, MaxDD, equity curve,
  per-symbol, per-regime attribution. Replay outcomes against any
  decision function.
- **Settings UI** — add/remove/enable/disable/pause symbols, edit aliases,
  configure alerts, manage blackouts, run backtests, halt trading, flatten
  all positions.
- **Paper / live modes** — swap via config or CLI flag.

---

## Quickstart

Python 3.12+ is required for the current `pandas-ta` dependency. Python 3.13 is the safest choice.

```bash
pip install -r requirements.txt
cp notifications.yaml.example notifications.yaml   # fill in your tokens
ollama pull gemma3:4b && ollama serve
python run.py --mode paper
```

Open http://localhost:8050 (dashboard) and http://localhost:8050/settings.

Full install notes in [docs/SETUP.md](docs/SETUP.md).

---

## Project layout

```
GemmaTraderAIBot/
├── README.md                   this file
├── CLAUDE.md                   rules for Claude AI sessions working on this repo
├── ROADMAP.md                  phased plan for profitability
├── LICENSE
├── pyproject.toml              package metadata (src layout)
├── requirements.txt
├── pytest.ini
├── Dockerfile, docker-compose.yml, .dockerignore
├── config.yaml                 core runtime config
├── symbols.yaml                broker alias registry (committed, seeded)
├── notifications.yaml.example  alerts template (copy to notifications.yaml)
├── news_blackouts.yaml.example blackout template
├── .env.example
│
├── run.py                      compatibility entrypoint
├── dashboard.py                compatibility entrypoint
├── server.py                   compatibility entrypoint
│
├── src/
│   └── gemma_trader/           main Python package
│       ├── run.py              entry point (Flask + trader + watchdog)
│       ├── dashboard.py        Flask routes + REST API
│       ├── server.py           webhook/API endpoints (legacy)
│       ├── local_trader.py     main polling loop
│       ├── gemma_analyzer.py   Ollama client + prompt
│       ├── risk_manager.py     pre-trade checks, sizing, outcome recording
│       ├── safety.py           kill-switch, drawdown breaker, heartbeat
│       ├── symbol_registry.py  generic → broker ticker mapping
│       ├── ensemble.py         multi-model gate + prompt hash + dedupe cache
│       ├── trade_reviewer.py   walk-forward pattern validator
│       ├── notifier.py         Telegram + Teams + WhatsApp
│       ├── news_calendar.py    blackout windows
│       ├── extra_features.py   funding / OB / BTC dom / correlation
│       ├── metrics.py          Sharpe/Sortino/MaxDD + attribution
│       ├── backtester.py       outcome replay
│       ├── broker_bridge.py    MT5 / Binance / Paper execution
│       └── mt5_data_feed.py    MT5 candles + ticks + positions
│
├── templates/                  HTML (dashboard + settings)
├── static/                     logos + icons
├── scripts/                    standalone utilities (create_doc, test harnesses)
├── tests/                      pytest suite (fast, no I/O)
├── docs/                       ARCHITECTURE / SETUP / CONFIG / API
└── logs/                       trades, decisions, outcomes (git-ignored)
```

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [CLAUDE.md](CLAUDE.md) | Rules for Claude AI sessions editing this repo |
| [ROADMAP.md](ROADMAP.md) | Phased profitability plan — read before going live |
| [docs/SETUP.md](docs/SETUP.md) | Install, first-run, Docker, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map, data flow, invariants |
| [docs/CONFIG.md](docs/CONFIG.md) | Every config file + key explained |
| [docs/API.md](docs/API.md) | Full HTTP + SocketIO reference |

---

## CLI

```bash
python run.py                   # defaults from config.yaml
python run.py --mode paper
python run.py --symbols BTCUSD ETHUSD   # use generic names from symbols.yaml
python run.py --port 8080
python run.py --no-trade         # dashboard only
```

---

## Self-learning loop

Every `adaptive.review_every_n_trades` trades, `trade_reviewer.py`
regenerates an adaptive context block that is prepended to Gemma's
system prompt. A pattern (e.g. "RSI oversold under HIGH_VOL regime on
SOLUSD") is promoted into the live prompt only if:

1. It has ≥ 30 samples on an older *train* slice.
2. It re-holds on a held-out *test* slice (≥ 10 samples).
3. It re-holds on the most recent window (≥ 15 samples).
4. Its combined sample clears a binomial significance test (p < 0.05).
5. Its direction (favour / avoid) agrees across all three slices.

Patterns that stop holding on the recent window are dropped automatically
— no lesson ossifies.

> The narrative "Gemma reviews itself" output is kept only as an advisory
> artifact (`logs/gemma_narrative_review.txt`) and is **not** auto-fed back
> into the live prompt. A 4B model cannot reliably critique its own trades.

Full config under `adaptive:` in `config.yaml`; see [docs/CONFIG.md](docs/CONFIG.md).

---

## Logs

All logs are written under `logs/` (git-ignored). See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#state--persistence) for the
full map.

---

## Tests

```bash
pytest                # 20 tests, ~0.5s, no network/MT5/Ollama required
```

---

## Disclaimer

Research / educational project. Automated trading carries substantial
risk of loss. **Paper-trade first** (see [ROADMAP.md](ROADMAP.md) Phase 0).
Use real capital only after you have validated sustained performance across
multiple market regimes. The authors accept no liability for trading losses.

## License

See [LICENSE](LICENSE).
