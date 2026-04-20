# Profitability Enhancement Suite

Six new modules (Phase 2) built to accelerate the path to profitable
live trading from 6 months to ~2 months.

| Module | Purpose | CLI |
|--------|---------|-----|
| `storage.py` | SQLite durable storage | `python -m gemma_trader.storage --migrate` |
| `historical_backtester.py` | Proper backtest on any date range | `python -m gemma_trader.historical_backtester --symbol BTC/USDT --days 90` |
| `advanced_features.py` | Microstructure + regime features | Imported by local_trader |
| `regime_detector.py` | HMM regime classification | Imported by local_trader |
| `ml_baseline.py` | XGBoost classifier alongside Gemma | `python -m gemma_trader.ml_baseline --train` |
| `hyperopt.py` | Bayesian hyperparameter tuning | `python -m gemma_trader.hyperopt --symbol BTC/USDT` |

---

## 1. Storage (SQLite)

**Replaces:** JSON file scheme that caps at ~500 outcomes.

### Migration
```bash
python -m gemma_trader.storage --migrate
```
Moves existing `logs/trade_outcomes.json` + `logs/gemma_decisions.json`
into `logs/trading.db`. JSON files are kept (dual-write mode); set
`storage.dual_write: false` in `config.yaml` once confident.

### Tables
- `trade_outcomes` — realized trades with indicators snapshot
- `gemma_decisions` — every Gemma call (even HOLDs)
- `trades` — open/closed position ledger
- `ohlcv_cache` — historical candles (24h TTL)
- `parameter_adjustments` — audit trail of threshold changes

### Query examples
```python
from gemma_trader.storage import get_db
db = get_db()
btc_wins = db.query_outcomes(symbol="BTCUSD", regime="trending_up")
db.count_outcomes(symbol="ETHUSD")
```

---

## 2. Historical Backtester

**Replaces:** Legacy `backtester.py` that only replays logged outcomes.

### Why it matters
The old backtester could only test what had already been traded. The new
one fetches historical OHLCV from Binance via ccxt and walks bar-by-bar
with slippage + commission modelling. You can test any prompt, any
strategy, on months of data in seconds.

### Example
```python
from gemma_trader.historical_backtester import HistoricalBacktester

def my_strategy(df, i, state):
    rsi = compute_rsi(df)
    if rsi < 30:
        return {"action": "BUY", "confidence": 0.8,
                "sl_distance_atr": 1.0, "tp_distance_atr": 1.5}
    return {"action": "HOLD", "confidence": 0.0}

bt = HistoricalBacktester()
result = bt.run(
    "BTC/USDT",
    start="2024-01-01", end="2024-04-01",
    strategy_fn=my_strategy,
    timeframe="1h",
    spread_pct=0.05, commission_pct=0.01,
)
print(f"Sharpe: {result.sharpe:.2f}, PF: {result.profit_factor:.2f}")
```

### Walk-forward validation
```python
results = bt.run_walk_forward(
    "BTC/USDT", "2024-01-01", "2024-06-01",
    my_strategy,
    train_days=30, test_days=7,
)
# List of BacktestResults, one per test window
```

---

## 3. Advanced Features

Features that are NOT in the standard 30+ indicator set, because alpha
lives in microstructure, not more momentum oscillators.

| Feature | Interpretation |
|---------|----------------|
| `order_flow_imbalance` | +1 = heavy buy pressure, -1 = heavy sell |
| `spread_atr_ratio` | > 0.25 = TP structurally unreachable; reject |
| `volume_profile_zscore` | > 1.5 = volume spike confirms breakout |
| `ret_autocorr_5` | > 0 = momentum regime; < 0 = mean-reversion |
| `realized_volatility_1h` | True vol, not ATR proxy |
| `hurst_exponent` | > 0.5 = trending; < 0.5 = mean-reverting |
| `btc_eth_spread_zscore` | Pairs-trade signal between BTC and ETH |
| `hour_of_day`, `session` | Time-of-day edge (Asia < London < NY) |
| `dist_from_60high_pct` | Pullback vs. breakout context |

Enable via `config.yaml`:
```yaml
advanced_features:
  enabled: true
  spread_filter: true
  max_spread_atr_ratio: 0.25
```

---

## 4. Regime Detector (HMM)

**Replaces:** Crude 3-bucket `vol_trend` classification.

Fits a Gaussian HMM to (returns, volatility, volume_zscore) and labels
each state post-hoc by cluster mean:

- `trending_up` — positive drift, low vol
- `trending_down` — negative drift, low vol
- `high_vol_chop` — high vol, no drift
- `low_vol_range` — low vol, no drift

### Usage
```python
from gemma_trader.regime_detector import get_regime_detector

detector = get_regime_detector()
detector.fit(historical_df)   # needs 500+ bars
detector.save("logs/hmm_model.pkl")

# Live
state = detector.current_state(live_df)
# {"state_id": 2, "label": "high_vol_chop", "probabilities": {...}}
```

If `hmmlearn` is not installed, falls back to rule-based classification
(3 states: low_vol_range, trending, high_vol_chop).

---

## 5. ML Baseline (XGBoost)

**Why:** If XGBoost matches a 9.6GB LLM at predicting trade outcomes,
your edge is in features, not model. This module quantifies that.

### Training
```bash
python -m gemma_trader.ml_baseline --train --min-samples 100
```

Trains walk-forward CV on `trade_outcomes` table. Outputs model to
`logs/ml_baseline.pkl` + metrics + top feature importances.

### Agreement gate
Once trained, Gemma + ML can be combined:

```python
from gemma_trader.ml_baseline import MLBaseline

baseline = MLBaseline()
baseline.load("logs/ml_baseline.pkl")

ml_pred = baseline.predict(market_data)  # {prob_win: 0.68}

allowed, reason = MLBaseline.agreement_gate(
    gemma_decision,
    ml_pred,
    min_ml_prob=0.55,
)
```

- Both BUY + ML prob_win > 0.55 → high-confidence trade
- Gemma BUY + ML prob_win < 0.55 → veto (logged)

Controlled by `config.yaml`:
```yaml
ml_baseline:
  enabled: true
  min_ml_prob: 0.55
  enforce_veto: false   # advisory-only first, then enforce
```

---

## 6. Bayesian Hyperopt

**Problem:** `confidence_threshold`, SL/TP multipliers, cooldown are
hand-tuned. Optimal values shift with regime.

**Solution:** Optuna-based search maximizing Sharpe on walk-forward
validation over the historical backtester.

### Run it
```bash
python -m gemma_trader.hyperopt \
    --symbol BTC/USDT \
    --days 90 \
    --timeframe 1h \
    --trials 100
```

Output: `config.optimized.yaml` that you can diff against `config.yaml`
and adopt manually (never auto-applied for safety).

### Parameters searched
- `confidence_threshold`: 0.50–0.85
- `sl_atr_multiplier`: 0.5–2.0
- `tp_atr_multiplier`: 1.0–3.0
- `risk_per_trade_pct`: 0.5–2.0

### Objective
```
score = Sharpe × (1 − min(MaxDD, 0.5))
```
Trades with < 5 total trades are penalized (overfitting guard).

---

## Recommended Usage Flow

### First week
```bash
# 1. Migrate existing data to SQLite
python -m gemma_trader.storage --migrate

# 2. Run historical backtest to see baseline edge
python -m gemma_trader.historical_backtester --symbol BTC/USDT --days 90

# 3. Let the bot trade paper for 14 days to accumulate outcomes
python run.py --mode paper
```

### Second week (after ~100 outcomes)
```bash
# 4. Train ML baseline on recent outcomes
python -m gemma_trader.ml_baseline --train

# 5. Run hyperopt to find tuned params
python -m gemma_trader.hyperopt --symbol BTC/USDT --trials 100

# 6. Diff config files and adopt selectively
diff config.yaml config.optimized.yaml
```

### Ongoing
- Retrain HMM weekly (every ~500 new bars)
- Retrain ML baseline every 50 new outcomes
- Rerun hyperopt monthly to catch regime shifts
- Review dashboard for Gemma/ML agreement rate — dropping rate = regime shifted

---

## Anti-Patterns (Do NOT Do)

- ❌ **Auto-apply hyperopt output** — always human-review first
- ❌ **Train ML on < 100 outcomes** — statistical noise
- ❌ **Enable ML enforce_veto before 2 weeks advisory** — need agreement stats first
- ❌ **Skip walk-forward validation** — in-sample Sharpe is not real Sharpe
- ❌ **Retrain on every new trade** — recipe for overfitting to noise
- ❌ **Ignore HMM disagreement with Gemma** — it's a regime-shift signal
