# Architecture

Rey Capital AI Bot is a single-process Python application that runs three
concurrent subsystems in one Flask process:

1. A **trading loop** (background thread in `run.py`) that polls MT5 every
   60s, asks Gemma for a decision, and places orders.
2. A **dashboard** (Flask + SocketIO) serving `/` and `/settings`.
3. A **watchdog** (background thread in `run.py`) that halts trading if the
   main loop stalls.

Everything is in-process; there is no message queue, no database. State
lives in YAML config files + JSON logs under `logs/`.

---

## Module map

```
                ┌──────────────────────────────────────────────┐
                │                   run.py                     │
                │    (entry point: Flask + trader + watchdog)  │
                └──────────────────┬───────────────────────────┘
                                   │
      ┌────────────────────────────┼─────────────────────────────┐
      │                            │                             │
      ▼                            ▼                             ▼
┌───────────────┐           ┌──────────────┐            ┌────────────────┐
│ dashboard.py  │           │ local_trader │            │  watchdog      │
│ (Flask app)   │           │ .run_cycle() │            │  thread        │
└───────┬───────┘           └──────┬───────┘            └───────┬────────┘
        │                          │                            │
        │                 ┌────────┼──────────┐                  │
        │                 │        │          │                  │
        │                 ▼        ▼          ▼                  │
        │           safety.py  news_cal  ensemble.py             │
        │           (halt?)    (blackout)(models+cache)          │
        │                                   │                    │
        │                                   ▼                    │
        │                         gemma_analyzer.py              │
        │                         (Ollama client + prompt)       │
        │                                   │                    │
        │                                   ▼                    │
        │                         risk_manager.py                │
        │                         (pre-trade checks + sizing)    │
        │                                   │                    │
        │                                   ▼                    │
        │                         broker_bridge.py  ──► MT5      │
        │                         (resolve via symbol_registry)  │
        │                                   │                    │
        │                                   ▼                    │
        │                         notifier.py ──► TG / Teams / WA│
        │                                                        │
        └────────────── safety.heartbeat() ──────────────────────┘
```

---

## Data flow (per cycle)

```
 1.  run_cycle()
 2.    safety.heartbeat()
 3.    reconnect MT5 if dropped  →  notifier("reconnect")
 4.    balance = broker.get_balance()
 5.    safety.update_equity(balance)   ──► trip breaker if dd > max
 6.    _check_closed_positions()       ──► outcomes + notifier("exit")
 7.    registry.load() (pick up UI changes)
 8.    for each active generic symbol:
 9.        if safety.is_halted() → skip
10.        if news.in_blackout() → skip
11.        df = mt5_feed.get_candles(generic)  ──► resolved to broker ticker
12.        indicators = calculate_indicators(df)
13.        decision = dedupe.lookup(indicators) or
                     ensemble_decide(indicators, config, analyze_with_gemma)
14.        risk_manager.can_trade(decision) → if False, skip
15.        pos_size = risk_manager.calculate_position_size(...)
16.        broker.place_order(generic, ...)  ──► resolved to broker ticker
17.        notifier("entry")
```

---

## State & persistence

| File                              | Owner                  | Purpose                              |
|-----------------------------------|------------------------|--------------------------------------|
| `config.yaml`                     | human-edited           | model, mode, risk, adaptive params  |
| `symbols.yaml`                    | `symbol_registry`      | generic → broker ticker aliases      |
| `notifications.yaml`              | `notifier`             | channel creds, event toggles         |
| `news_blackouts.yaml`             | `news_calendar`        | UTC blackout windows                 |
| `logs/trades.json`                | `risk_manager`         | every open request (audit)           |
| `logs/trade_outcomes.json`        | `risk_manager`         | closed trades with indicator snap   |
| `logs/gemma_decisions.json`       | `local_trader`         | raw Gemma outputs + prompt hash     |
| `logs/adaptive_context.txt`       | `trade_reviewer`       | validated patterns fed into prompt  |
| `logs/gemma_narrative_review.txt` | `trade_reviewer`       | advisory only — not live            |
| `logs/parameter_adjustments.json` | `risk_manager`         | threshold change audit trail        |
| `logs/safety_state.json`          | `safety`               | halt state + equity peak (resumes)  |
| `logs/trade_journal.json`         | `local_trader`         | per-trade reasoning + user comments |
| `logs/lot_overrides.json`         | `dashboard`            | manual lot-size overrides per sym   |

Everything under `logs/` is git-ignored.

---

## Thread model

| Thread                 | Started by    | What it does                                 |
|------------------------|---------------|----------------------------------------------|
| MainThread (Flask)     | `run.py`      | SocketIO + HTTP; never blocks                |
| `trader_loop`          | `run.py`      | sleeps poll_interval, calls `run_cycle`      |
| `watchdog_loop`        | `run.py`      | sleeps poll_interval, checks heartbeat age   |
| SocketIO background    | Flask-SocketIO| per-client emit buffers                      |

All three write to shared state (`safety.state`, `registry.symbols`,
`risk_manager.open_trades`). Contention is low because the trader loop
runs once a minute. Mutexes are inside `safety` and `symbol_registry`.

---

## Invariants

1. **Symbol resolution happens only at the broker/feed boundary.** Upstream
   code (trader, risk, reviewer, metrics) uses generic names. See CLAUDE.md
   §8.
2. **The bot only touches positions with `magic == 240411`.** Any other
   open position on the MT5 account is invisible to the bot.
3. **`safety.is_halted()` gates every new entry.** There is no code path
   that places an order without checking this flag.
4. **The LLM never sees private keys or account balances.** `gemma_analyzer`
   only ships indicators + adaptive context.
5. **`logs/trade_outcomes.json` is append-only in practice.** The reviewer
   treats it as ground truth; the risk manager bounds its growth by
   keeping the last 500 entries.

---

## Where to look when something breaks

| Symptom                                          | File to start in          |
|--------------------------------------------------|---------------------------|
| "No trades today"                                | `local_trader.analyze_symbol` — check halt/news/conf gates |
| "Trades on the wrong ticker" (e.g. `GOLD` not `XAUUSD`) | `symbol_registry.resolve` call site |
| "MT5 says not connected"                         | `mt5_data_feed._connect` + terminal running? |
| "Dashboard shows stale balance"                  | `dashboard.api_health` — MT5 read fails silently |
| "Alerts never arrive"                            | `notifier` + `notifications.yaml` enabled flags |
| "Same pattern lesson appears and disappears"     | `trade_reviewer` — check sample counts for freshness |
| "Bot keeps opening redundant positions"          | `risk_manager.can_trade` — cooldown logic |
| "Drawdown breaker won't reset"                   | `logs/safety_state.json` — delete `breaker_tripped`/`halted` fields |
