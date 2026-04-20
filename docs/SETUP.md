# Setup

## Prerequisites

- Python 3.12+ (3.13 recommended)
- Windows 10/11 for live MT5 (the MetaTrader5 Python bindings are
  Windows-only). Paper mode + dashboard work fine on macOS/Linux.
- [Ollama](https://ollama.com) running locally with a Gemma model pulled
- MetaTrader 5 terminal installed and logged in to your broker

## Install

```bash
git clone <this-repo>
cd GemmaTraderAIBot
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you are using Python 3.14 and see pip reject `pandas-ta`, recreate the virtualenv with Python 3.13. The project depends on the beta `pandas-ta` releases currently published on PyPI.

On macOS/Linux the `MetaTrader5` line in `requirements.txt` will fail —
that is expected. Install everything else:

```bash
grep -v -i "^metatrader5" requirements.txt > /tmp/req.txt
pip install -r /tmp/req.txt
```

## Pull the model

```bash
ollama pull gemma3:4b        # minimum
ollama pull gemma3:12b       # recommended for ensemble gate
ollama serve                 # leave running
```

## First-run config

1. Copy the example files:
   ```bash
   cp notifications.yaml.example notifications.yaml
   cp news_blackouts.yaml.example news_blackouts.yaml
   ```
2. Edit `config.yaml`:
   - `trading.mode`: `paper` for first 2 weeks (mandatory — see ROADMAP.md)
   - `ollama.model`: the tag you pulled above
   - `broker.mt5.login`/`password`/`server`: optional if MT5 terminal
     is already logged in
3. Edit `symbols.yaml` — the seeded file already has BTC/ETH/LTC/XRP/SOL
   and a disabled `GOLD` example showing two broker aliases.

## Start

```bash
python run.py
```

Dashboard: http://localhost:8050
Settings:  http://localhost:8050/settings

## Command-line flags

```bash
python run.py --mode paper            # force paper mode (overrides config)
python run.py --symbols BTCUSD ETHUSD # trade only these (generics)
python run.py --port 8080             # custom port
python run.py --no-trade              # dashboard only, useful for backtesting
```

## Verifying the install

```bash
pytest                    # should be 20/20 green
```

## Docker

```bash
docker compose up --build
```

This starts Ollama + bot in the same network. The bot runs in paper mode
and mounts `logs/`, `config.yaml`, `symbols.yaml`, `notifications.yaml`,
and `news_blackouts.yaml` from the host so your state survives restarts.

Pull the model *inside* the Ollama container the first time:

```bash
docker exec -it rey-ollama ollama pull gemma3:4b
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: MetaTrader5` on macOS/Linux | Use paper mode or Docker; the CFD broker path needs Windows |
| Gemma responses are empty or timeout | `ollama serve` not running, or model not pulled; check `http://localhost:11434/api/tags` |
| Dashboard loads but no trades | Check `/api/safety/status` — likely halted; check confidence threshold in `/api/health` |
| Alerts don't arrive | Go to `/settings` → Alerts → Test entry, watch the JSON response |
| MT5 "symbol not found" | The symbol is not on your broker's server; edit `symbols.yaml` aliases |
