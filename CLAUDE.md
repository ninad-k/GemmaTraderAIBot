# CLAUDE.md — Rules for working on this repo

You are working on **Rey Capital AI Bot** (aka GemmaTraderAIBot), an LLM-driven
crypto/CFD scalper. This file tells future Claude sessions how to collaborate on
this codebase effectively. Read it before starting any non-trivial task.

---

## 1. What this project is

A self-learning trading bot:
- **Data**: MT5 1-minute candles (+ TradingView fallback)
- **Brain**: Google Gemma 3 via local Ollama (4B or 12B)
- **Execution**: MT5 for CFDs, Binance for spot crypto, Paper for sim
- **Self-learning**: walk-forward validated pattern library fed back into prompt
- **UI**: Flask + SocketIO dashboard at `:8050`, plus `/settings` page

The single most important rule about this project: **it handles real money**.
A bug here is not a lint warning, it is a realised loss. Optimise for safety
and auditability over cleverness.

---

## 2. Non-negotiable rules

1. **Never loosen a risk check to "make tests pass"**. If `risk_manager.py`
   refuses a trade, that refusal is a feature. If the test was wrong, fix the
   test.
2. **Never remove the kill-switch gate** in `local_trader.analyze_symbol`.
3. **Never silently swallow broker errors**. `broker.place_order` returning
   `{"status": "error", ...}` must propagate — never coerce to `filled`.
4. **Never commit secrets**. `notifications.yaml`, `config.yaml` with real
   MT5 creds, any API key — all are git-ignored. If you see one staged, stop
   and tell the user.
5. **Never modify `logs/trade_outcomes.json` by hand**. The self-learning loop
   trusts its contents as ground truth. Treat it like a database.
6. **Never change the magic number `240411`**. It's how MT5 knows which
   positions this bot owns. Changing it orphans live positions.
7. **Never auto-feed the Gemma self-narrative review into live prompts**.
   A 4B model cannot reliably critique its own trades. The narrative stays
   advisory (`logs/gemma_narrative_review.txt`). Only the statistically
   validated `adaptive_context.txt` is live.

---

## 3. Architecture at a glance

```
run.py                    Entry point — starts Flask + trader thread + watchdog
├── dashboard.py          Flask app + REST API + SocketIO
├── local_trader.py       The polling loop: feed → indicators → Gemma → broker
│   ├── gemma_analyzer.py Ollama client + prompt builder + parser
│   ├── ensemble.py       Multi-model gate + prompt hash + dedupe cache
│   ├── risk_manager.py   Pre-trade checks, position sizing, outcome recording
│   ├── safety.py         Kill-switch + drawdown breaker + heartbeat
│   ├── trade_reviewer.py Walk-forward pattern validator (self-learning)
│   ├── symbol_registry.py generic → broker ticker mapping
│   ├── notifier.py       Telegram + Teams + WhatsApp alerts
│   ├── news_calendar.py  Blackout windows
│   ├── extra_features.py Funding / OB imbalance / BTC dom / correlation
│   ├── broker_bridge.py  MT5 / Binance / Paper execution
│   └── mt5_data_feed.py  MT5 candle + tick feed
├── metrics.py            Sharpe / Sortino / MaxDD / attribution
├── backtester.py         Replay outcomes vs. an arbitrary decision fn
└── templates/settings.html  Settings UI (symbols, alerts, news, safety, metrics)
```

See `docs/ARCHITECTURE.md` for data flow diagrams and invariants.

---

## 4. How to make changes safely

### Before you write any code

- Run `pytest` (from repo root, it only picks up `tests/`) and confirm 20/20
  green. If it's not green before your change, stop and report.
- Read the file you're about to modify **completely**, not just the function.
  These modules are < 1200 lines each and often have subtle coupling.

### When adding a feature

- Prefer **a new self-contained module** over editing a 1000-line file. New
  modules are the pattern used by Phase 2-6 (`safety.py`, `notifier.py`,
  `metrics.py`, `ensemble.py`, etc).
- Add a pytest file under `tests/` for every new module. Tests must not
  require MT5, Ollama, a network, or a real broker. Use `monkeypatch` and
  `tmp_path` — there are clean examples in `tests/test_safety.py` and
  `tests/test_metrics_and_backtest.py`.
- Wire new features **opt-in**. Default off. Example: the ensemble is only
  active if `config.yaml` has `ensemble.enabled: true`. This keeps existing
  deployments unchanged.

### When modifying a risk rule

- Write the test first. It should cover (a) the rule firing, (b) the rule
  *not* firing in the adjacent case, (c) the parameter boundary.
- Log the adjustment to `logs/parameter_adjustments.json` so the audit trail
  survives.

### When touching the prompt

- Bump the prompt version and let `ensemble.prompt_hash` tag decisions with
  the new hash automatically.
- Run `backtester.run_backtest` against the last 500 outcomes with the old
  and new prompts before committing. A prompt that loses on historical data
  will lose on future data.

### When touching self-learning

- Walk-forward validation in `trade_reviewer.py` is the reason the bot
  doesn't chase noise. **Do not reduce** `min_train_samples`,
  `min_test_samples`, `freshness_min_samples`, or `significance_alpha`
  without a written justification. If you want a faster adaptation loop,
  the correct answer is more data, not weaker gates.

---

## 5. Testing discipline

- `pytest` must be green before every commit.
- The test suite runs in ~0.5s. If you introduce a slow test (>1s), move it
  behind `@pytest.mark.slow` and document it.
- Do not import MetaTrader5 or ccxt unconditionally in new modules — they
  make the test suite fail on non-Windows CI. Wrap imports in a try/except
  or guard them behind `if TYPE_CHECKING`.
- The legacy `test_gemma.py` and `test_lots.py` at the repo root are **not**
  pytest files. They live in `scripts/` and are ad-hoc REPL scripts. Don't
  let pytest pick them up — `pytest.ini` restricts to `tests/`.

---

## 6. How to extend the settings UI

All settings endpoints live in `dashboard.py` under `/api/settings/...` and
all persist to a YAML file next to the code:

| Domain        | Endpoint                          | File                    |
|---------------|-----------------------------------|-------------------------|
| Symbols       | `/api/settings/symbols`           | `symbols.yaml`          |
| Notifications | `/api/settings/notifications`     | `notifications.yaml`    |
| News blackout | `/api/settings/news`              | `news_blackouts.yaml`   |
| Active broker | `/api/settings/active_broker`     | `symbols.yaml`          |

When adding a new domain, copy the pattern:
1. A small singleton class with `load()` / `save()` in its own module.
2. Four routes in `dashboard.py`: GET (state), POST (upsert), DELETE (remove),
   optionally a POST `…/test` or `…/toggle`.
3. A new card in `templates/settings.html` with a `loadX()`/`saveX()`
   function and a call to `loadX()` at the bottom of the script block.

---

## 7. How to extend the trader pipeline

The analysis pipeline in `local_trader.analyze_symbol` is:

```
safety.is_halted?  →  news.in_blackout?  →  fetch candles  →
indicators  →  dedupe cache?  →  ensemble_decide(gemma)  →
risk_manager.can_trade?  →  position sizing  →  broker.place_order  →
register_trade  →  notifier.notify("entry")
```

New gates go **before** the LLM call (to save tokens). New features go
into the indicators dict (everything downstream expects a flat dict).
New models go into `ensemble.ensemble_decide` — do not fork the pipeline.

---

## 8. Broker symbol aliasing (important subtlety)

Generic symbols (`GOLD`, `BTCUSD`) live everywhere in the code. Only the
two edge modules — `broker_bridge.py` and `mt5_data_feed.py` — call
`symbol_registry.get_registry().resolve(symbol)` to translate to the broker's
ticker (`XAUUSD`, `XAUUSD_`, etc). **Never resolve upstream of those modules**
or you will end up with broker-specific tickers in logs, outcomes, and the
self-learning pattern library, breaking cross-broker portability.

---

## 9. When in doubt

- Prefer small reversible edits over sweeping refactors.
- Ask the user before destructive ops (rm -rf, force push, DB drops).
- If a test fails in a way you don't understand, stop and read. Don't
  "fix" it by changing the assertion.
- If you find a bug unrelated to your current task, flag it — don't
  silently bundle it into your change.
