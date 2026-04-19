# HTTP API reference

All endpoints live under the Flask app started by `run.py`, default
`http://localhost:8050`.

## Health & state

| Method | Path                         | Purpose                              |
|--------|------------------------------|--------------------------------------|
| GET    | `/api/health`                | mode, model, balance, symbols, threshold |
| GET    | `/api/decisions`             | full Gemma decision log              |
| GET    | `/api/trades`                | all trades (open + closed)           |
| GET    | `/api/stats`                 | decision breakdown + per-symbol stats |
| GET    | `/api/symbols`               | latest decision per symbol           |
| GET    | `/api/learning`              | win rate, adaptive context, adjustments |
| GET    | `/api/pnl`                   | real-time unrealised PnL from MT5    |
| GET    | `/api/journal`               | trade journal with user comments     |
| POST   | `/api/journal/comment`       | add comment `{trade_id, comment}`    |
| GET    | `/api/gemma_thinking`        | last 20 raw Gemma responses          |
| POST   | `/api/lot_override`          | set `{symbol, lot_size}`             |
| GET    | `/api/lot_overrides`         | current overrides                    |

## Safety

| Method | Path                    | Purpose                                  |
|--------|-------------------------|------------------------------------------|
| GET    | `/api/safety/status`    | halted?, reason, equity, drawdown, heartbeat |
| POST   | `/api/safety/halt`      | `{reason}` — stop new entries           |
| POST   | `/api/safety/resume`    | resume trading                           |
| POST   | `/api/safety/flatten`   | close all bot-owned positions (requires trader attached) |

## Metrics & backtest

| Method | Path                            | Purpose                          |
|--------|---------------------------------|----------------------------------|
| GET    | `/api/metrics/summary`          | Sharpe/Sortino/MaxDD/win rate   |
| GET    | `/api/metrics/equity_curve`     | cumulative equity points         |
| GET    | `/api/metrics/per_symbol`       | per-symbol PnL + Sharpe          |
| GET    | `/api/metrics/per_regime`       | per-regime attribution           |
| POST   | `/api/backtest`                 | `{threshold, start_balance, limit}` — replay outcomes |

Example:
```bash
curl -X POST http://localhost:8050/api/backtest \
  -H 'Content-Type: application/json' \
  -d '{"threshold": 0.65, "start_balance": 100000}'
```

## Settings — symbols

| Method | Path                                      | Purpose                        |
|--------|-------------------------------------------|--------------------------------|
| GET    | `/api/settings/symbols`                   | registry state                 |
| POST   | `/api/settings/symbols`                   | upsert `{generic, aliases, enabled, active}` |
| DELETE | `/api/settings/symbols/<generic>`         | remove                         |
| POST   | `/api/settings/symbols/<generic>/toggle`  | `{enabled?, active?}`          |
| POST   | `/api/settings/active_broker`             | `{broker}`                     |

## Settings — notifications

| Method | Path                                  | Purpose                             |
|--------|---------------------------------------|-------------------------------------|
| GET    | `/api/settings/notifications`         | full config                         |
| POST   | `/api/settings/notifications`         | overwrite — `{channels, events}`    |
| POST   | `/api/settings/notifications/test`    | `{event}` — fire a test message     |

## Settings — news

| Method | Path                               | Purpose                        |
|--------|------------------------------------|--------------------------------|
| GET    | `/api/settings/news`               | list windows                   |
| POST   | `/api/settings/news`               | `{start, end, label}` — add    |
| DELETE | `/api/settings/news/<index>`       | remove by index                |

## SocketIO events

| Event           | Payload                                      | Emitted when                |
|-----------------|----------------------------------------------|-----------------------------|
| `new_decision`  | `{symbol, close, decision, timestamp}`       | Gemma returns a decision    |
| `new_trade`     | trade dict                                   | order filled                |
| `trade_closed`  | `{symbol, profit, timestamp}`                | position closed on MT5      |
| `stats_update`  | `{cycle, balance, open_trades, daily_pnl}`   | each cycle, + reviewer pass |
