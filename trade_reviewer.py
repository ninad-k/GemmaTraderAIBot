"""
Rey Capital AI Bot — Trade Reviewer (Self-Improving Logic)
============================================================
Analyzes trade outcomes and produces an adaptive context block that is
prepended to Gemma's system prompt.

Design goals (and what this module deliberately does NOT do):
  - Walk-forward validation: patterns are derived from an older "train"
    slice and must re-hold on a newer "test" slice before being promoted
    into live advice. Prevents in-sample overfitting.
  - Statistical significance: each pattern must clear a binomial test
    against a 50% baseline (configurable alpha). Prevents flagging noise
    from 3-trade samples as signal.
  - Regime scoping: patterns are bucketed by (trend_regime, vol_regime)
    so a lesson from a trending high-vol week does not leak into a
    ranging low-vol week.
  - Freshness decay: a pattern that was valid historically must still
    hold on a recent window or it is dropped from live advice.
  - Consistent windows: threshold adjustment and freshness checks use
    the SAME recent window. No mixing "all-time" with "last 20".
  - The narrative self-review by Gemma is kept for human inspection
    only — it is not auto-fed back into the live prompt, because a 4B
    model cannot reliably critique itself.
"""

import json
import logging
from datetime import datetime
from math import comb
from pathlib import Path

from gemma_analyzer import review_trades_with_gemma

logger = logging.getLogger("trade_reviewer")


def _binomial_two_tailed_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-tailed binomial p-value for observing k wins in n trades."""
    if n <= 0:
        return 1.0
    probs = [comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    observed = probs[k]
    return sum(pi for pi in probs if pi <= observed + 1e-12)


def _classify_regime(snap: dict) -> tuple[str, str]:
    """
    Derive (trend_regime, vol_regime) from an indicators snapshot.
    Kept intentionally coarse — fine-grained buckets split samples too thin.
    """
    trend = (snap.get("trend") or "UNKNOWN").upper()
    adx = snap.get("adx")
    try:
        adx_v = float(adx) if adx is not None else None
    except (TypeError, ValueError):
        adx_v = None

    if trend in ("UP", "BULLISH") and adx_v is not None and adx_v >= 25:
        trend_regime = "STRONG_UP"
    elif trend in ("DOWN", "BEARISH") and adx_v is not None and adx_v >= 25:
        trend_regime = "STRONG_DOWN"
    elif adx_v is not None and adx_v < 20:
        trend_regime = "RANGING"
    else:
        trend_regime = "WEAK_TREND"

    vol_ratio = snap.get("vol_ratio")
    try:
        vr = float(vol_ratio) if vol_ratio is not None else None
    except (TypeError, ValueError):
        vr = None

    if vr is None:
        vol_regime = "VOL_UNKNOWN"
    elif vr >= 1.5:
        vol_regime = "HIGH_VOL"
    elif vr <= 0.7:
        vol_regime = "LOW_VOL"
    else:
        vol_regime = "NORMAL_VOL"

    return trend_regime, vol_regime


def _pattern_key_values(snap: dict) -> dict:
    """Extract pattern-worthy dimensions from an indicator snapshot."""
    rsi = snap.get("rsi")
    try:
        rsi_v = float(rsi) if rsi is not None else None
    except (TypeError, ValueError):
        rsi_v = None

    if rsi_v is None:
        rsi_bucket = None
    elif rsi_v < 30:
        rsi_bucket = "RSI_oversold"
    elif rsi_v > 70:
        rsi_bucket = "RSI_overbought"
    else:
        rsi_bucket = "RSI_neutral"

    return {
        "rsi": rsi_bucket,
        "trend": snap.get("trend"),
        "ichimoku": snap.get("ichimoku_signal"),
        "volume": snap.get("vol_trend"),
    }


class TradeReviewer:
    def __init__(self, config: dict, risk_manager=None):
        self.config = config
        self.risk_manager = risk_manager
        self.adaptive_cfg = config.get("adaptive", {})

        log_cfg = config.get("logging", {})
        self.outcome_path = Path(log_cfg.get("outcome_log", "logs/trade_outcomes.json"))
        self.adaptive_ctx_path = Path(log_cfg.get("adaptive_context", "logs/adaptive_context.txt"))
        self.param_adj_path = Path(log_cfg.get("parameter_adjustments", "logs/parameter_adjustments.json"))
        self.narrative_review_path = self.adaptive_ctx_path.parent / "gemma_narrative_review.txt"

        self.last_review_count = 0
        self.last_weekly_review = None

    # ─── Public API ───

    def analyze_performance(self, force: bool = False) -> dict:
        outcomes = self._load_outcomes()
        if not outcomes:
            return {}

        review_every = self.adaptive_cfg.get("review_every_n_trades", 50)
        if not force and len(outcomes) - self.last_review_count < review_every:
            return {}

        min_adapt = self.adaptive_cfg.get("min_trades_for_adaptation", 30)
        if len(outcomes) < min_adapt:
            logger.info(
                f"Analysis skipped: {len(outcomes)} trades < {min_adapt} required"
            )
            return self._basic_summary(outcomes)

        self.last_review_count = len(outcomes)
        logger.info(f"Analyzing performance: {len(outcomes)} trades")

        summary = self._overall_stats(outcomes)

        # Recent-window stats drive both threshold adjustment AND freshness.
        window = self.adaptive_cfg.get("analysis_window", 100)
        recent = outcomes[-window:]

        validated_patterns = self._derive_validated_patterns(outcomes, recent)

        context = self._build_adaptive_context(summary, validated_patterns, recent)
        self._save_adaptive_context(context)

        if self.risk_manager:
            recent_wins = sum(1 for o in recent if o.get("profit", 0) > 0)
            recent_wr = (recent_wins / len(recent) * 100) if recent else 0
            self.risk_manager.adjust_threshold(recent_wr, len(recent))

        logger.info(
            f"Performance: WR={summary['win_rate']:.1f}% | "
            f"PF={summary['profit_factor']:.2f} | "
            f"Total PnL=${summary['total_pnl']:.2f} | "
            f"validated patterns={len(validated_patterns)}"
        )
        return summary

    def weekly_review(self, force: bool = False) -> str:
        """
        Narrative self-review by Gemma. Stored for human inspection only —
        it is NOT fed back into the live system prompt. Small models cannot
        reliably critique themselves; their narratives hallucinate edges.
        """
        if not self.adaptive_cfg.get("weekly_review", False) and not force:
            return ""

        now = datetime.now()
        if not force and self.last_weekly_review:
            days_since = (now - self.last_weekly_review).days
            if days_since < 7:
                return ""

        outcomes = self._load_outcomes()
        if len(outcomes) < 50:
            return ""

        logger.info("Running weekly Gemma narrative review (advisory only)...")
        self.last_weekly_review = now

        lessons = review_trades_with_gemma(outcomes[-50:], self.config)
        if lessons:
            try:
                self.narrative_review_path.parent.mkdir(parents=True, exist_ok=True)
                stamp = now.strftime("%Y-%m-%d %H:%M")
                self.narrative_review_path.write_text(
                    f"[{stamp}] Gemma narrative self-review (advisory, not live):\n\n{lessons}"
                )
            except Exception as e:
                logger.error(f"Failed to save narrative review: {e}")
        return lessons

    def get_performance_summary(self) -> dict:
        outcomes = self._load_outcomes()
        if not outcomes:
            return {"total": 0, "win_rate": 0, "total_pnl": 0}
        wins = sum(1 for o in outcomes if o.get("profit", 0) > 0)
        total = len(outcomes)
        return {
            "total": total,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "total_pnl": round(sum(o.get("profit", 0) for o in outcomes), 2),
        }

    # ─── Stats ───

    def _basic_summary(self, outcomes: list) -> dict:
        wins = [o for o in outcomes if o.get("profit", 0) > 0]
        total = len(outcomes)
        return {
            "total_trades": total,
            "win_rate": round(len(wins) / total * 100, 1) if total else 0,
            "total_pnl": round(sum(o.get("profit", 0) for o in outcomes), 2),
            "timestamp": datetime.now().isoformat(),
            "note": "below min_trades_for_adaptation — no pattern lessons generated",
        }

    def _overall_stats(self, outcomes: list) -> dict:
        wins = [o for o in outcomes if o.get("profit", 0) > 0]
        losses = [o for o in outcomes if o.get("profit", 0) <= 0]
        total = len(outcomes)
        win_rate = len(wins) / total * 100 if total else 0
        avg_win = sum(o["profit"] for o in wins) / len(wins) if wins else 0
        avg_loss = sum(o["profit"] for o in losses) / len(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss else float("inf")
        total_pnl = sum(o.get("profit", 0) for o in outcomes)

        return {
            "total_trades": total,
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "timestamp": datetime.now().isoformat(),
        }

    # ─── Walk-forward validated pattern derivation ───

    def _derive_validated_patterns(self, outcomes: list, recent: list) -> list:
        """
        1. Split outcomes chronologically into train / test (older vs newer).
        2. For each (regime, dimension, bucket) compute win rate on train & test.
        3. Keep only patterns that:
              - Have >= min_train_samples on train and >= min_test_samples on test.
              - Clear binomial significance on the combined (train+test) sample.
              - Point the SAME direction on train and test (both favor, or both avoid).
              - Also hold on the fresh `recent` window (>= freshness_min_samples).
        """
        train_frac = self.adaptive_cfg.get("train_fraction", 0.7)
        min_train = self.adaptive_cfg.get("min_train_samples", 30)
        min_test = self.adaptive_cfg.get("min_test_samples", 10)
        min_fresh = self.adaptive_cfg.get("freshness_min_samples", 15)
        alpha = self.adaptive_cfg.get("significance_alpha", 0.05)
        fav_wr = self.adaptive_cfg.get("favorable_wr", 0.60)
        avoid_wr = self.adaptive_cfg.get("avoid_wr", 0.40)
        regime_scope = self.adaptive_cfg.get("regime_scoping", True)

        split = int(len(outcomes) * train_frac)
        if split < min_train:
            return []
        train = outcomes[:split]
        test = outcomes[split:]
        if len(test) < min_test:
            return []

        def group(slice_):
            """Returns {(regime, dim, bucket): [is_win, ...]}"""
            buckets: dict = {}
            for o in slice_:
                snap = o.get("indicators_snapshot", {}) or {}
                trend_r, vol_r = _classify_regime(snap)
                regime = f"{trend_r}|{vol_r}" if regime_scope else "ALL"
                is_win = o.get("profit", 0) > 0
                for dim, val in _pattern_key_values(snap).items():
                    if val is None or str(val).upper() == "UNKNOWN":
                        continue
                    key = (regime, dim, str(val))
                    buckets.setdefault(key, []).append(is_win)
            return buckets

        train_buckets = group(train)
        test_buckets = group(test)
        fresh_buckets = group(recent)

        validated = []
        for key, train_results in train_buckets.items():
            if len(train_results) < min_train:
                continue
            test_results = test_buckets.get(key, [])
            if len(test_results) < min_test:
                continue
            fresh_results = fresh_buckets.get(key, [])
            if len(fresh_results) < min_fresh:
                continue

            def wr(results):
                return sum(results) / len(results) if results else 0

            train_wr = wr(train_results)
            test_wr = wr(test_results)
            fresh_wr = wr(fresh_results)

            is_fav = train_wr >= fav_wr and test_wr >= fav_wr and fresh_wr >= fav_wr
            is_avoid = train_wr <= avoid_wr and test_wr <= avoid_wr and fresh_wr <= avoid_wr
            if not (is_fav or is_avoid):
                continue

            combined = train_results + test_results
            wins = sum(combined)
            p_val = _binomial_two_tailed_p(wins, len(combined))
            if p_val >= alpha:
                continue

            regime, dim, bucket = key
            validated.append({
                "regime": regime,
                "dimension": dim,
                "bucket": bucket,
                "direction": "FAVOR" if is_fav else "AVOID",
                "train_wr": round(train_wr * 100, 1),
                "test_wr": round(test_wr * 100, 1),
                "fresh_wr": round(fresh_wr * 100, 1),
                "train_n": len(train_results),
                "test_n": len(test_results),
                "fresh_n": len(fresh_results),
                "p_value": round(p_val, 4),
            })

        validated.sort(
            key=lambda x: (x["p_value"], -abs(x["fresh_wr"] - 50))
        )
        return validated[:12]

    # ─── Context rendering ───

    def _build_adaptive_context(self, summary: dict, patterns: list, recent: list) -> str:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"LESSONS FROM VALIDATED TRADE PATTERNS (auto-updated {stamp})",
            f"Scope: {summary['total_trades']} total trades | "
            f"WR={summary['win_rate']:.1f}% | "
            f"PF={summary['profit_factor']:.2f} | "
            f"Total PnL=${summary['total_pnl']:.2f}",
            "",
            "Each lesson below was derived from an older train slice, "
            "re-confirmed on a held-out newer slice, re-confirmed again on "
            "the most recent window, and cleared a binomial significance test.",
            "Lessons are scoped to a market regime — apply only when the "
            "current regime matches.",
            "",
        ]

        recent_wins = sum(1 for o in recent if o.get("profit", 0) > 0)
        recent_wr = (recent_wins / len(recent) * 100) if recent else 0
        lines.append(
            f"Recent window ({len(recent)} trades): WR={recent_wr:.1f}%"
        )

        if not patterns:
            lines.append("")
            lines.append(
                "No patterns have yet cleared validation. Trade with base "
                "rules only — do not infer edges from small samples."
            )
            return "\n".join(lines)

        lines.append("")
        lines.append("VALIDATED PATTERNS:")
        for p in patterns:
            lines.append(
                f"- [{p['regime']}] {p['direction']} when "
                f"{p['dimension']}={p['bucket']} — "
                f"WR train {p['train_wr']}% (n={p['train_n']}), "
                f"test {p['test_wr']}% (n={p['test_n']}), "
                f"fresh {p['fresh_wr']}% (n={p['fresh_n']}), "
                f"p={p['p_value']}"
            )

        return "\n".join(lines)

    # ─── IO ───

    def _load_outcomes(self) -> list:
        try:
            if self.outcome_path.exists():
                text = self.outcome_path.read_text(encoding="utf-8-sig").strip()
                if text:
                    return json.loads(text)
        except Exception as e:
            logger.error(f"Failed to load outcomes: {e}")
        return []

    def _save_adaptive_context(self, context: str):
        try:
            self.adaptive_ctx_path.parent.mkdir(parents=True, exist_ok=True)
            self.adaptive_ctx_path.write_text(context)
            logger.debug(f"Adaptive context saved ({len(context)} chars)")
        except Exception as e:
            logger.error(f"Failed to save adaptive context: {e}")
