"""
Rey Capital AI Bot — Risk Manager
====================================
Validates trades against risk rules, tracks outcomes,
manages per-symbol cooldowns, and dynamically adjusts
confidence thresholds based on performance.
"""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: dict):
        self.config = config
        self.trading_cfg = config["trading"]
        self.risk_cfg = config["risk_management"]
        self.adaptive_cfg = config.get("adaptive", {})
        self.daily_pnl = 0.0
        self.open_trades = []
        self.trade_date = date.today()

        # Per-symbol cooldown tracking
        self.last_trade_time = {}       # symbol → datetime
        self.symbol_streaks = {}        # symbol → {"type": "win"|"loss", "count": int}
        self.cooled_down_symbols = {}   # symbol → datetime (cooldown expires)

        # Dynamic threshold
        self.original_threshold = self.trading_cfg["confidence_threshold"]
        self.current_threshold = self.original_threshold

        # Log paths
        log_cfg = config.get("logging", {})
        self.trade_log_path = Path(log_cfg.get("trade_log", "logs/trades.json"))
        self.outcome_log_path = Path(log_cfg.get("outcome_log", "logs/trade_outcomes.json"))
        self.param_adj_path = Path(log_cfg.get("parameter_adjustments", "logs/parameter_adjustments.json"))
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Core Risk Checks ───

    def can_trade(self, decision: dict, market_data: dict) -> tuple[bool, str]:
        """
        Check if a trade is allowed under current risk rules.
        Returns (allowed, reason).
        """
        # Reset daily PnL if new day
        if date.today() != self.trade_date:
            self.daily_pnl = 0.0
            self.trade_date = date.today()

        symbol = decision.get("symbol", "UNKNOWN")

        # 1. Check allowed symbols
        if symbol not in self.trading_cfg["allowed_symbols"]:
            return False, f"{symbol} not in allowed symbols list"

        # 2. Check confidence threshold (use dynamic threshold)
        if decision["confidence"] < self.current_threshold:
            return False, (
                f"Confidence {decision['confidence']:.2f} below "
                f"threshold {self.current_threshold:.2f}"
            )

        # 3. Check max open trades
        if len(self.open_trades) >= self.trading_cfg["max_open_trades"]:
            return False, f"Max open trades reached ({self.trading_cfg['max_open_trades']})"

        # 4. Check if already in a trade for this symbol
        if any(t["symbol"] == symbol for t in self.open_trades):
            return False, f"Already have an open trade for {symbol}"

        # 5. Check daily loss limit
        if self.daily_pnl <= -self.risk_cfg["max_daily_loss_pct"]:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}%"

        # 6. Per-symbol cooldown (min time between trades on same symbol)
        cooldown_min = self.trading_cfg.get("cooldown_minutes", 5)
        last_time = self.last_trade_time.get(symbol)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds() / 60
            if elapsed < cooldown_min:
                remaining = cooldown_min - elapsed
                return False, f"Cooldown active for {symbol}: {remaining:.1f}min remaining"

        # 7. Streak-based cooldown
        if symbol in self.cooled_down_symbols:
            cooldown_expires = self.cooled_down_symbols[symbol]
            if datetime.now() < cooldown_expires:
                remaining = (cooldown_expires - datetime.now()).total_seconds() / 60
                return False, f"{symbol} on loss-streak cooldown: {remaining:.0f}min remaining"
            else:
                del self.cooled_down_symbols[symbol]

        return True, "approved"

    # ─── Position Sizing ───

    def calculate_position_size(self, account_balance: float, atr: float,
                                sl_atr_mult: float, symbol: str = "") -> dict:
        """
        Calculate position size in MT5 lots based on ATR and risk %.

        Uses MT5 symbol info (contract size, tick value, min/max lot, lot step)
        to calculate proper lot size. Falls back to simple calc if MT5 unavailable.
        """
        risk_pct = self.trading_cfg["max_position_size_pct"] / 100
        risk_amount = account_balance * risk_pct
        sl_distance = atr * sl_atr_mult

        if sl_distance <= 0:
            return {"qty": 0, "risk_amount": 0, "sl_distance": 0}

        # Try to get MT5 symbol info for proper lot sizing
        lot_size = self._calc_mt5_lots(symbol, risk_amount, sl_distance)

        if lot_size <= 0:
            # Fallback: simple calculation (for paper mode)
            lot_size = round(risk_amount / sl_distance, 6)

        return {
            "qty": lot_size,
            "risk_amount": round(risk_amount, 2),
            "sl_distance": round(sl_distance, 2),
        }

    def _calc_mt5_lots(self, symbol: str, risk_amount: float,
                       sl_distance: float) -> float:
        """
        Calculate lot size using MT5 symbol specifications.

        Formula: lots = risk_amount / (sl_distance / tick_size * tick_value)
        Then clamp to min_lot/max_lot and round to lot_step.
        """
        if not symbol:
            return 0

        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            if not info:
                return 0

            tick_size = info.trade_tick_size
            tick_value = info.trade_tick_value
            min_lot = info.volume_min
            max_lot = info.volume_max
            lot_step = info.volume_step

            if tick_size <= 0 or tick_value <= 0:
                return 0

            # How many ticks in our SL distance
            ticks_in_sl = sl_distance / tick_size

            # Value of SL per 1 lot
            sl_value_per_lot = ticks_in_sl * tick_value

            if sl_value_per_lot <= 0:
                return 0

            # Raw lot size
            lots = risk_amount / sl_value_per_lot

            # Clamp to min/max
            lots = max(min_lot, min(max_lot, lots))

            # Round to lot step
            lots = round(lots / lot_step) * lot_step
            lots = round(lots, 3)  # avoid floating point issues

            logger.debug(
                f"Lot calc: {symbol} risk=${risk_amount:.2f} "
                f"sl_dist={sl_distance:.4f} → {lots} lots "
                f"(min={min_lot}, max={max_lot}, step={lot_step})"
            )

            return lots

        except ImportError:
            return 0
        except Exception as e:
            logger.warning(f"MT5 lot calc failed for {symbol}: {e}")
            return 0

    # ─── Trade Lifecycle ───

    def register_trade(self, trade: dict):
        """Register an open trade and update cooldown timer."""
        self.open_trades.append(trade)
        self.last_trade_time[trade["symbol"]] = datetime.now()
        self._log_trade(trade)
        logger.info(
            f"Trade registered: {trade['action']} {trade['symbol']} "
            f"qty={trade['qty']}"
        )

    def close_trade(self, symbol: str, pnl: float):
        """Close a trade and update daily PnL."""
        self.open_trades = [t for t in self.open_trades if t["symbol"] != symbol]
        self.daily_pnl += pnl
        logger.info(
            f"Trade closed: {symbol} PnL={pnl:.2f}% | "
            f"Daily PnL={self.daily_pnl:.2f}%"
        )

    # ─── Outcome Tracking (Self-Learning) ───

    def record_outcome(self, trade: dict, close_price: float, profit: float):
        """
        Record a trade outcome for self-learning.
        Stores full context: entry indicators, Gemma reasoning, and result.
        """
        outcome = {
            "symbol": trade.get("symbol"),
            "action": trade.get("action"),
            "entry_price": trade.get("entry_price"),
            "close_price": close_price,
            "sl": trade.get("sl"),
            "tp": trade.get("tp"),
            "qty": trade.get("qty"),
            "profit": round(profit, 2),
            "result": "WIN" if profit > 0 else "LOSS",
            "confidence": trade.get("confidence"),
            "reason": trade.get("reason"),
            "indicators_snapshot": trade.get("indicators_snapshot", {}),
            "entry_time": trade.get("timestamp"),
            "close_time": datetime.now().isoformat(),
            "duration_minutes": self._calc_duration(trade.get("timestamp")),
        }

        # Save to outcome log
        try:
            outcomes = []
            if self.outcome_log_path.exists():
                text = self.outcome_log_path.read_text(encoding="utf-8-sig").strip()
                if text:
                    outcomes = json.loads(text)
            outcomes.append(outcome)
            self.outcome_log_path.write_text(json.dumps(outcomes[-500:], indent=2))
        except Exception as e:
            logger.error(f"Failed to log outcome: {e}")

        # Update symbol streak
        self._update_streak(trade["symbol"], profit > 0)

        # Update daily PnL
        self.daily_pnl += profit

        logger.info(
            f"Outcome recorded: {trade['symbol']} "
            f"{'WIN' if profit > 0 else 'LOSS'} ${profit:.2f}"
        )

        return outcome

    def _update_streak(self, symbol: str, is_win: bool):
        """Track win/loss streaks per symbol and trigger cooldowns."""
        if symbol not in self.symbol_streaks:
            self.symbol_streaks[symbol] = {"type": "win" if is_win else "loss", "count": 0}

        streak = self.symbol_streaks[symbol]
        current_type = "win" if is_win else "loss"

        if streak["type"] == current_type:
            streak["count"] += 1
        else:
            streak["type"] = current_type
            streak["count"] = 1

        # Check if loss streak triggers cooldown
        cooldown_trigger = self.adaptive_cfg.get("cooldown_on_streak_loss", 3)
        cooldown_duration = self.adaptive_cfg.get("cooldown_duration_minutes", 30)

        if streak["type"] == "loss" and streak["count"] >= cooldown_trigger:
            self.cooled_down_symbols[symbol] = (
                datetime.now() + timedelta(minutes=cooldown_duration)
            )
            logger.warning(
                f"{symbol}: {streak['count']} consecutive losses — "
                f"cooldown {cooldown_duration}min"
            )

    def get_symbol_streak(self, symbol: str) -> dict:
        """Get current win/loss streak for a symbol."""
        return self.symbol_streaks.get(symbol, {"type": "none", "count": 0})

    def get_all_streaks(self) -> dict:
        """Get all symbol streaks."""
        return dict(self.symbol_streaks)

    # ─── Dynamic Threshold Adjustment ───

    def adjust_threshold(self, win_rate: float, total_trades: int):
        """
        Auto-adjust confidence threshold based on recent win rate.
        - Win rate < 40% → raise threshold (be more selective)
        - Win rate > 60% → lower threshold (take more trades)
        """
        if not self.adaptive_cfg.get("enabled", False):
            return

        min_trades = self.adaptive_cfg.get("min_trades_for_adaptation", 5)
        if total_trades < min_trades:
            return

        max_thresh = self.adaptive_cfg.get("max_confidence_threshold", 0.85)
        min_thresh = self.adaptive_cfg.get("min_confidence_threshold", 0.50)

        old_threshold = self.current_threshold

        if win_rate < 40:
            # Losing — be more selective
            self.current_threshold = min(
                self.current_threshold + 0.05, max_thresh
            )
        elif win_rate > 60:
            # Winning — can be slightly less selective
            self.current_threshold = max(
                self.current_threshold - 0.02, min_thresh
            )
        # Between 40-60%: no change

        if self.current_threshold != old_threshold:
            adjustment = {
                "timestamp": datetime.now().isoformat(),
                "type": "confidence_threshold",
                "old_value": round(old_threshold, 3),
                "new_value": round(self.current_threshold, 3),
                "win_rate": round(win_rate, 1),
                "total_trades": total_trades,
                "description": (
                    f"Threshold {'raised' if self.current_threshold > old_threshold else 'lowered'} "
                    f"from {old_threshold:.2f} to {self.current_threshold:.2f} "
                    f"(win rate: {win_rate:.1f}%)"
                ),
            }

            self._log_adjustment(adjustment)
            logger.info(
                f"Threshold adjusted: {old_threshold:.2f} -> "
                f"{self.current_threshold:.2f} (win rate: {win_rate:.1f}%)"
            )

    def _log_adjustment(self, adjustment: dict):
        """Log parameter adjustment."""
        try:
            adjs = []
            if self.param_adj_path.exists():
                text = self.param_adj_path.read_text(encoding="utf-8-sig").strip()
                if text:
                    adjs = json.loads(text)
            adjs.append(adjustment)
            self.param_adj_path.write_text(json.dumps(adjs[-100:], indent=2))
        except Exception as e:
            logger.error(f"Failed to log adjustment: {e}")

    # ─── Helpers ───

    def _calc_duration(self, entry_time_str: str) -> float:
        """Calculate trade duration in minutes."""
        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            return round((datetime.now() - entry_time).total_seconds() / 60, 1)
        except Exception:
            return 0

    def _log_trade(self, trade: dict):
        """Append trade to JSON log file."""
        try:
            trades = []
            if self.trade_log_path.exists():
                text = self.trade_log_path.read_text(encoding="utf-8-sig").strip()
                if text:
                    trades = json.loads(text)
            trades.append(trade)
            self.trade_log_path.write_text(json.dumps(trades, indent=2))
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")


