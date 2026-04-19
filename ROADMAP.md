# Profitability Roadmap

How to take the Gemma 4 trader from "interesting prototype" to "consistently
profitable on risk-adjusted terms". This is written as a phased plan. Each
phase has **exit criteria** — don't move to the next phase until the previous
one is stably green.

> **Premise.** Most retail LLM traders lose money not because the model is
> bad, but because (a) their costs/spreads eat their edge, (b) they can't
> measure whether they actually have edge, and (c) they over-fit to the
> last 30 days of market regime. This plan attacks those three failure
> modes first.

---

## Phase 0 — Honest baseline (Week 1-2)

**Goal: know exactly how bad/good the current bot is before changing anything.**

1. Run in **paper mode** for 14 calendar days, all 5 seeded symbols.
2. Keep the prompt, model, and thresholds frozen — resist the urge to tune.
3. At end of window, compute from `logs/trade_outcomes.json`:
   - Total trades (aim for ≥100 per symbol, else extend)
   - Win rate, average win, average loss, profit factor
   - Sharpe, Sortino, Max drawdown (use `metrics.summary()`)
   - **Break-even cost analysis**: `(spread + commission + slippage) × trades`
     vs. total PnL. If costs > 40% of gross PnL, no amount of model tuning
     will help — go to Phase 1.

**Exit criteria**: profit factor ≥ 1.0 on paper *after* modelled costs.
If not, the bot is not yet worth running live; continue phases.

---

## Phase 1 — Cost control (Week 3-4)

The #1 killer of scalpers is transaction cost. Before any model work,
squeeze this.

1. **Tighter entry gates**. Raise `confidence_threshold` to 0.70+ until
   trade count drops 40-60%. The goal is fewer, higher-quality trades.
2. **Feature-dedupe cache** (already built in `ensemble.FeatureDedupeCache`).
   Turn it on with TTL = 60s. Target: 15-25% hit rate, which cuts Ollama
   load and duplicate trade attempts.
3. **Spread filter** (new work): before entry, require `spread_atr_ratio < 0.25`.
   A spread > 25% of ATR means the TP is structurally unreachable.
4. **Cooldown after every trade, not just losses**. Crypto 1M bars are
   noisy; two trades 90s apart on the same symbol correlates outcomes.

**Exit criteria**: cost-per-trade (spread + commission in account currency)
< 12% of average win.

---

## Phase 2 — Measurement before model (Week 5-6)

You cannot optimise what you cannot see. Before touching the model, build
honest attribution.

1. **Per-regime PnL** (already in `/api/metrics/per_regime`). Split outcomes
   by volatility regime (HIGH / NORMAL / LOW) and directionality (trend vs.
   chop). If the bot has edge in only one regime, the fix is a regime
   filter, not a better model.
2. **Per-hour / per-weekday PnL**. Crypto liquidity varies dramatically.
   Typical finding: Asia overnight produces negative edge, London open
   produces positive edge. Add blackout windows for losing hours in
   `news_blackouts.yaml`.
3. **Per-pattern PnL**. The `trade_reviewer.py` walk-forward already
   groups by `signal_type + regime`. Surface the top-3 and bottom-3
   patterns on the dashboard.
4. **Decision latency tracking**. Timestamp: candle-close → Gemma response
   → broker ack. If end-to-end > 8s on a 60s bar, the signal is stale
   before it executes. Either batch parallel Ollama calls or upgrade
   hardware / model.

**Exit criteria**: you can answer "where does the edge come from?" in one
sentence, pointing at data.

---

## Phase 3 — Data quality (Week 7-8)

More *relevant* data beats a bigger model.

1. **Enable extra features** (already built in `extra_features.py`):
   - Funding rate (Binance)
   - Order-book imbalance (top 20 levels)
   - BTC dominance (CoinGecko global)
   - Correlation guard (don't pile into 3 correlated longs)
2. **Multi-timeframe context**. Feed 1m candles for execution but show
   the model 5m and 15m trend direction. Cheap and moves win rate.
3. **Session/regime features**. Hour-of-day, day-of-week, VIX-equivalent
   (BTC ATR percentile), as explicit inputs to Gemma.
4. **Outcome labels** for self-learning. Enrich the snapshot in
   `record_outcome` with the features above so `trade_reviewer.py`
   can build regime-specific pattern lessons.

**Exit criteria**: validated (p<0.05, train/test/fresh agreement) pattern
library contains ≥ 8 non-redundant rules across regimes.

---

## Phase 4 — Model discipline (Week 9-12)

Only now is it worth touching the model.

1. **Ensemble gate** (already built). Run 4B + 12B side by side via
   `ensemble.ensemble_decide`. Require both to agree for BUY/SELL, else
   HOLD. Expect trade count to drop ~50% and win rate to rise.
2. **Prompt versioning**. Every decision is tagged with `prompt_hash` already.
   Any prompt change must ship with a 500-outcome `run_backtest()` result
   in the commit message. No prompt change without that.
3. **Specialised prompts per regime**. The generic prompt tries to do
   trend + mean-reversion + momentum. Split into `trend_prompt.md`,
   `meanrev_prompt.md`, `chop_prompt.md` and route via the detected regime.
4. **Temperature discipline**. Keep `temperature: 0.1` or lower. Creativity
   is not what you want from a trading decision.

**Exit criteria**: out-of-sample Sharpe (on held-out last 4 weeks, after
training on older data) ≥ 1.2 on paper.

---

## Phase 5 — Risk-adjusted position sizing (Week 13-14)

The Kelly insight: with positive edge, size by confidence; without, don't
trade. The current sizer is a flat 1% per trade — leaving edge on the table
when signal quality is high, and taking too much when it's marginal.

1. **Confidence-scaled size**: `qty = base_qty × (confidence - 0.5) × 2`.
   High-conf trades risk 1.5x, marginal trades risk 0.3x.
2. **Kelly fraction with safety**: use the per-pattern validated win rate
   and avg win/loss to compute Kelly, then size at 0.25× Kelly (full Kelly
   is too volatile for anyone psychologically).
3. **Correlation-adjusted total exposure**: cap simultaneous correlated
   exposure to 2× single-position size (use the correlation guard in
   `extra_features.correlation_ok`).

**Exit criteria**: Sharpe improves without MaxDD degrading.

---

## Phase 6 — Controlled live deployment (Week 15-20)

Only after all five previous exit criteria met. **Do not skip**.

1. **Micro-live**. Min lot size, single symbol (pick the best one from
   per-symbol attribution), 1% account risk cap, drawdown breaker at 5%.
2. **Parallel paper**: keep a paper account running the *same* config side
   by side. If live and paper diverge by >20% over 100 trades, your slippage/
   fill model is wrong — investigate before scaling.
3. **Two-week smoke window** before scaling to multiple symbols or raising
   position size. No exceptions for "obvious" wins.
4. **Weekly review ritual** (human, not LLM):
   - Top 3 winners — are they edge or variance?
   - Top 3 losers — pattern or isolated?
   - Did any rule fire unexpectedly?
   - Did costs land inside the modelled budget?
5. **Monthly prompt review**. Regenerate `adaptive_context.txt` from
   the last 90 days. If the validated pattern list changed by >30%,
   the regime has shifted — reconsider the whole phase 3-4 output.

**Exit criteria**: 6 consecutive weeks of positive risk-adjusted PnL
on live micro-lot.

---

## Phase 7 — Scale carefully (ongoing)

1. **Add symbols one at a time**. Each new symbol resets the 6-week clock.
2. **Add capital in steps of 2x**. Keep drawdown breaker pct absolute,
   so actual dollar drawdown remains proportional.
3. **Keep infra boring**. Docker-compose (already built), the same
   watchdog, the same notifier. Don't chase microservices glory.
4. **Automated regime detection**. When BTC ATR percentile crosses
   > 90 (high vol) or < 10 (dead vol), shrink size by 50% automatically.

---

## Anti-patterns to avoid forever

- **Over-fitting to the recent 100 trades**. The walk-forward validator
  exists exactly to prevent this. Don't bypass it.
- **Adding a feature because "it feels right"**. Every feature must pay
  rent in measurable Sharpe improvement on held-out data.
- **Raising leverage to recover drawdown**. This is the retail death spiral.
  Drawdown breaker at 10% is *non-negotiable*.
- **Trusting model confidence alone**. LLMs can be confidently wrong.
  The risk-manager gates exist because confidence is necessary but not
  sufficient.
- **Running a new prompt live before backtesting it**. See Phase 4.
- **Using the LLM to critique its own trades for live feedback**. A 4B
  model cannot reliably do that. Keep the narrative advisory only.

---

## Kill criteria (know when to stop)

If after 3 full phases (≈6 weeks) of honest work you cannot reach Phase 0
exit criteria on paper, **stop trading this strategy**. The cost of
continuing is real even in paper — you're anchoring your expectations and
eating engineering hours that could fund a better approach. Redirect to:
- longer timeframes (1h scalper → 1d swing)
- different instruments (crypto 24/7 is actually harder than FX/index
  because there's no session structure to hide behind)
- or shelving the LLM approach in favour of classical ML with the
  pattern library as features.

Profitability is a scientific claim. Treat it like one.
