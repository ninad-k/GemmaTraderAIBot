"""
Broker Bridge
Executes trades on MT5, Binance, or logs paper trades.
"""

import logging
from datetime import datetime
from abc import ABC, abstractmethod

from gemma_trader.symbol_registry import get_registry

logger = logging.getLogger(__name__)


def _resolve(symbol: str) -> str:
    """Map a generic symbol to the current active broker's ticker."""
    try:
        return get_registry().resolve(symbol)
    except Exception:
        return symbol


class BaseBroker(ABC):
    @abstractmethod
    def get_balance(self) -> float:
        pass

    @abstractmethod
    def place_order(self, symbol: str, action: str, qty: float,
                    sl: float, tp: float) -> dict:
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> dict:
        pass


class PaperBroker(BaseBroker):
    """Simulated broker for paper trading."""

    def __init__(self, initial_balance: float = 100_000):
        self.balance = initial_balance
        self.positions = {}
        self.order_history = []
        logger.info(f"Paper broker initialized with balance: {self.balance}")

    def get_balance(self) -> float:
        return self.balance

    def place_order(self, symbol: str, action: str, qty: float,
                    sl: float, tp: float) -> dict:
        symbol = _resolve(symbol)
        order = {
            "order_id": f"PAPER-{len(self.order_history) + 1:04d}",
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "sl": sl,
            "tp": tp,
            "timestamp": datetime.now().isoformat(),
            "status": "filled"
        }
        self.positions[symbol] = order
        self.order_history.append(order)
        logger.info(f"[PAPER] [PAPER] {action} {qty} {symbol} | SL={sl} TP={tp}")
        return order

    def close_position(self, symbol: str) -> dict:
        symbol = _resolve(symbol)
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            logger.info(f"[PAPER] [PAPER] Closed {symbol}")
            return {"status": "closed", "position": pos}
        return {"status": "no_position"}


class MT5Broker(BaseBroker):
    """MetaTrader 5 broker connection."""

    def __init__(self, config: dict):
        self.config = config["broker"]["mt5"]
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5

            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return

            if self.config["login"]:
                authorized = mt5.login(
                    login=self.config["login"],
                    password=self.config["password"],
                    server=self.config["server"]
                )
                if not authorized:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return

            self.connected = True
            info = mt5.account_info()
            logger.info(f"MT5 connected | Account: {info.login} | Balance: {info.balance}")

        except ImportError:
            logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")

    def get_balance(self) -> float:
        if not self.connected:
            return 0.0
        info = self.mt5.account_info()
        return info.balance if info else 0.0

    def _get_filling_mode(self, symbol: str):
        """Auto-detect the correct filling mode for a symbol."""
        info = self.mt5.symbol_info(symbol)
        if info is None:
            return self.mt5.ORDER_FILLING_IOC

        filling = info.filling_mode
        # filling_mode is a bitmask: bit 0 = FOK, bit 1 = IOC, bit 2 = RETURN
        if filling & 1:  # FOK supported
            return self.mt5.ORDER_FILLING_FOK
        elif filling & 2:  # IOC supported
            return self.mt5.ORDER_FILLING_IOC
        else:  # RETURN / default
            return self.mt5.ORDER_FILLING_RETURN

    def place_order(self, symbol: str, action: str, qty: float,
                    sl: float, tp: float) -> dict:
        if not self.connected:
            logger.error("MT5 not connected")
            return {"status": "error", "reason": "not connected"}

        symbol = _resolve(symbol)
        order_type = self.mt5.ORDER_TYPE_BUY if action == "BUY" else self.mt5.ORDER_TYPE_SELL
        price_func = self.mt5.symbol_info_tick(symbol)

        if not price_func:
            return {"status": "error", "reason": f"Cannot get price for {symbol}"}

        price = price_func.ask if action == "BUY" else price_func.bid
        filling_mode = self._get_filling_mode(symbol)

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": qty,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 240411,
            "comment": "gemma4-trader",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        logger.info(f"Sending order: {symbol} {action} {qty} @ {price} | SL={sl} TP={tp} | filling={filling_mode}")
        result = self.mt5.order_send(request)
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            logger.error(f"MT5 order failed: {result.retcode} — {result.comment}")
            # Retry with FOK if IOC failed
            if result.retcode == 10030:
                for alt_fill in [self.mt5.ORDER_FILLING_FOK, self.mt5.ORDER_FILLING_RETURN]:
                    if alt_fill == filling_mode:
                        continue
                    request["type_filling"] = alt_fill
                    logger.info(f"Retrying with filling mode {alt_fill}")
                    result = self.mt5.order_send(request)
                    if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                        break
            if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(f"MT5 order FINAL fail: {result.retcode} — {result.comment}")
                return {"status": "error", "retcode": result.retcode, "comment": result.comment}

        logger.info(f"[OK] MT5 {action} {qty} {symbol} @ {price} | SL={sl} TP={tp}")
        return {
            "status": "filled",
            "order_id": result.order,
            "price": price,
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "sl": sl,
            "tp": tp,
        }

    def close_position(self, symbol: str) -> dict:
        if not self.connected:
            return {"status": "error", "reason": "not connected"}

        symbol = _resolve(symbol)
        positions = self.mt5.positions_get(symbol=symbol)
        if not positions:
            return {"status": "no_position"}

        # Magic-number isolation: only close positions owned by this bot.
        # Foreign positions (other strategies, manual trades) are ignored.
        own_positions = [p for p in positions if int(getattr(p, "magic", 0)) == 240411]
        if not own_positions:
            logger.info(f"No bot-owned position for {symbol} (magic=240411)")
            return {"status": "no_position"}

        for pos in own_positions:
            close_type = self.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.ORDER_TYPE_BUY
            tick = self.mt5.symbol_info_tick(symbol)
            price = tick.bid if pos.type == 0 else tick.ask

            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "magic": 240411,
                "comment": "gemma4-close",
            }
            result = self.mt5.order_send(request)
            logger.info(f"Closed position {pos.ticket}: {result.retcode}")

        return {"status": "closed"}


class BinanceBroker(BaseBroker):
    """Binance broker connection."""

    def __init__(self, config: dict):
        self.config = config["broker"]["binance"]
        self.exchange = None
        self._connect()

    def _connect(self):
        try:
            import ccxt
            self.exchange = ccxt.binance({
                "apiKey": self.config["api_key"],
                "secret": self.config["api_secret"],
                "enableRateLimit": True,
                "options": {"defaultType": "future"}  # use futures
            })
            balance = self.exchange.fetch_balance()
            logger.info(f"Binance connected | USDT Balance: {balance['USDT']['free']}")
        except ImportError:
            logger.error("ccxt not installed. Run: pip install ccxt")
        except Exception as e:
            logger.error(f"Binance connection error: {e}")

    def get_balance(self) -> float:
        if not self.exchange:
            return 0.0
        balance = self.exchange.fetch_balance()
        return float(balance["USDT"]["free"])

    def place_order(self, symbol: str, action: str, qty: float,
                    sl: float, tp: float) -> dict:
        if not self.exchange:
            return {"status": "error", "reason": "not connected"}

        symbol = _resolve(symbol)
        try:
            side = "buy" if action == "BUY" else "sell"
            order = self.exchange.create_market_order(symbol, side, qty)

            # Set SL/TP as separate orders
            sl_side = "sell" if action == "BUY" else "buy"
            self.exchange.create_order(symbol, "stop_market", sl_side, qty,
                                        params={"stopPrice": sl})
            self.exchange.create_order(symbol, "take_profit_market", sl_side, qty,
                                        params={"stopPrice": tp})

            logger.info(f"[OK] Binance {action} {qty} {symbol} | SL={sl} TP={tp}")
            return {"status": "filled", "order": order}

        except Exception as e:
            logger.error(f"Binance order failed: {e}")
            return {"status": "error", "reason": str(e)}

    def close_position(self, symbol: str) -> dict:
        if not self.exchange:
            return {"status": "error"}
        symbol = _resolve(symbol)
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if float(pos["contracts"]) > 0:
                    side = "sell" if pos["side"] == "long" else "buy"
                    self.exchange.create_market_order(symbol, side, pos["contracts"],
                                                      params={"reduceOnly": True})
            return {"status": "closed"}
        except Exception as e:
            return {"status": "error", "reason": str(e)}


def create_broker(config: dict) -> BaseBroker:
    """Factory to create the right broker based on config."""
    if config["trading"]["mode"] == "paper":
        return PaperBroker()

    broker_name = config["broker"]["name"]
    if broker_name == "mt5":
        return MT5Broker(config)
    elif broker_name == "binance":
        return BinanceBroker(config)
    else:
        logger.warning(f"Unknown broker '{broker_name}', falling back to paper")
        return PaperBroker()



