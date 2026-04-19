# Configuration reference

All configuration lives in YAML files at the repo root. Secrets
(`notifications.yaml`) are git-ignored; version-controlled files
(`config.yaml`, `symbols.yaml`) should NOT contain credentials.

---

## `config.yaml` — core runtime

| Key                                          | Purpose                                         |
|----------------------------------------------|-------------------------------------------------|
| `server.host` / `server.port`                | Flask bind address                              |
| `data_source`                                | `mt5` (uses MT5 feed, falls back to TradingView) |
| `mt5_data.timeframe`                         | Candle TF (`1m`, `5m`, ...)                     |
| `mt5_data.n_bars`                            | Bars to pull per cycle (500 is plenty)          |
| `mt5_data.poll_interval_seconds`             | Cycle cadence (60 = one 1m bar per run)         |
| `ollama.url` / `ollama.model`                | Local Ollama endpoint + model tag               |
| `ollama.temperature` / `ollama.num_predict`  | Generation params (keep temperature ≤ 0.2)      |
| `trading.mode`                               | `paper` or `live`                               |
| `trading.confidence_threshold`               | Static floor; risk_manager may raise it dynamically |
| `trading.max_position_size_pct`              | % of equity risked per trade                    |
| `trading.max_open_trades`                    | Concurrent position cap                         |
| `trading.cooldown_minutes`                   | Min minutes between trades on same symbol       |
| `trading.allowed_symbols`                    | Legacy; the registry (`symbols.yaml`) takes precedence |
| `risk_management.stop_loss_atr_multiplier`   | SL = this × ATR                                 |
| `risk_management.take_profit_atr_multiplier` | TP = this × ATR                                 |
| `risk_management.max_daily_loss_pct`         | Daily PnL floor; soft block (new trades only)  |
| `risk_management.max_drawdown_pct`           | Hard circuit breaker; halts & flattens on trip  |
| `broker.name`                                | `mt5` or `binance`                              |
| `broker.mt5.magic`                           | Must stay `240411` — see CLAUDE.md §2           |
| `adaptive.*`                                 | Self-learning tuning — see ROADMAP §Phase 3-4   |

### Optional: ensemble mode

Not present in the shipped config. Add it to enable multi-model gating:

```yaml
ensemble:
  enabled: true
  models: ["gemma3:4b", "gemma3:12b"]
  min_agreement: 2          # both must return the same action
  dedupe_ttl: 60            # seconds; 0 disables dedupe cache
```

When enabled, `ensemble_decide` replaces the single-model call. Measure
outcome with `backtester.run_backtest()` before/after.

---

## `symbols.yaml` — broker aliasing

The registry maps **generic** symbols (what the trader, risk manager,
outcomes, and pattern library all use internally) to **broker-specific
tickers** (what the broker actually recognises). Switching brokers is
a one-line change: `active_broker`.

```yaml
active_broker: ic_markets
symbols:
  - generic: GOLD
    enabled: true         # honoured by the UI; if false, never considered
    active: true          # enabled but paused? set this to false
    aliases:
      ic_markets: XAUUSD
      cfi: XAUUSD_
      oanda: XAU_USD
```

- `enabled: false` removes the symbol from every list until re-enabled.
- `active: false` keeps the symbol enabled (still shown in UI) but
  the trader loop skips it.
- Unknown generic (not in `symbols`) passes through unchanged — useful
  for ad-hoc CLI overrides.

Manage entirely from `/settings` — or edit the YAML directly and the
registry will pick up changes on the next cycle (the trader calls
`registry.load()` each run).

---

## `notifications.yaml` — alerts (git-ignored)

See `notifications.yaml.example` for the full shape.

```yaml
channels:
  telegram:
    enabled: true
    bot_token: "123456:ABC..."
    chat_id: "-1001234567890"
  teams:
    enabled: false
    webhook_url: "https://outlook.office.com/webhook/..."
  whatsapp:
    enabled: false
    phone_number_id: "123456789012345"
    access_token: "EAAB..."
    to: "+15555551234"
events:
  entry: true
  exit: true
  halt: true
  breaker: true
  reconnect: true
  resume: true
```

### WhatsApp (Meta Cloud API) quickstart

1. Create a Meta Business app with WhatsApp product enabled.
2. From Meta for Developers → WhatsApp → API setup, copy:
   - **Phone number ID** → `phone_number_id`
   - **Temporary access token** (24h) or generate a permanent system-user
     token → `access_token`
3. Add your own phone as a test recipient, verify it.
4. Set `to` to your number in E.164 (`+15555551234`).
5. Go to `/settings` → Alerts → "Test entry" to fire a message.

For production, replace the 24-hour token with a permanent system-user
token — Meta's 24h tokens expire silently and break alerts.

---

## `news_blackouts.yaml` — UTC blackout windows

```yaml
windows:
  - start: "2026-04-19T14:00:00"
    end:   "2026-04-19T14:30:00"
    label: "FOMC minutes"
```

During a blackout window the trader:
- **skips new entries** for all symbols
- **continues to manage existing positions** (SL/TP untouched)

Times are UTC. Manage via `/settings` → News Blackouts.

---

## Environment variables

Only one is read today; everything else comes from YAML:

| Var      | Purpose                              |
|----------|--------------------------------------|
| `PORT`   | Fallback dashboard port (8050)       |

See `.env.example` for future additions.

---

## Precedence

```
CLI flag (run.py)   >   config.yaml   >   code default
symbols.yaml (registry)   >   config.yaml:trading.allowed_symbols
```
