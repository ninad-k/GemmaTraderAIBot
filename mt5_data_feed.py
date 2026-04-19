"""
MT5 Data Feed Module
=====================
Fetches live candle data directly from MetaTrader 5 terminal.
Separate from broker_bridge.py which handles order execution.

Requirements:
    - MetaTrader 5 desktop must be running
    - pip install MetaTrader5
"""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger("mt5_data_feed")

try:
    import MetaTrader5 as mt5

    TIMEFRAME_MAP = {
        "1m": mt5.TIMEFRAME_M1,
        "2m": mt5.TIMEFRAME_M2,
        "3m": mt5.TIMEFRAME_M3,
        "5m": mt5.TIMEFRAME_M5,
        "10m": mt5.TIMEFRAME_M10,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "2h": mt5.TIMEFRAME_H2,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
        "1w": mt5.TIMEFRAME_W1,
        "1M": mt5.TIMEFRAME_MN1,
    }
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    TIMEFRAME_MAP = {}
    logger.warning("MetaTrader5 package not installed. Run: pip install MetaTrader5")


class MT5DataFeed:
    """
    Fetches candle data and tick prices from MetaTrader 5.
    Uses the same MT5 connection credentials from config.yaml.
    """

    def __init__(self, config: dict):
        if not MT5_AVAILABLE:
            raise ImportError(
                "MetaTrader5 package not installed. "
                "Run: pip install MetaTrader5"
            )

        self.config = config
        self.mt5_config = config.get("broker", {}).get("mt5", {})
        self.connected = False
        self._connect()

    def _connect(self):
        """Initialize MT5 and login."""
        try:
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 initialize failed: {error}")
                logger.error(
                    "Make sure MetaTrader 5 desktop is running!"
                )
                return

            # Login if credentials provided
            login = self.mt5_config.get("login", 0)
            if login and login != 0:
                authorized = mt5.login(
                    login=int(login),
                    password=str(self.mt5_config.get("password", "")),
                    server=str(self.mt5_config.get("server", "")),
                )
                if not authorized:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return

            self.connected = True
            account = mt5.account_info()
            if account:
                logger.info(
                    f"MT5 Data Feed connected | "
                    f"Account: {account.login} | "
                    f"Server: {account.server} | "
                    f"Balance: {account.balance}"
                )
            else:
                logger.info("MT5 Data Feed connected (no account info)")

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")

    def ensure_symbol(self, symbol: str) -> bool:
        """
        Ensure a symbol is visible in Market Watch.
        MT5 requires symbols to be selected before fetching data.
        """
        if not self.connected:
            return False

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol '{symbol}' not found in MT5")
            return False

        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol '{symbol}' in Market Watch")
                return False
            logger.info(f"Added '{symbol}' to Market Watch")

        return True

    def get_candles(self, symbol: str, timeframe_str: str,
                    n_bars: int = 200) -> pd.DataFrame:
        """
        Fetch OHLCV candle data from MT5.

        Args:
            symbol: MT5 symbol name (e.g., "US100_Spot", "XAUUSD_")
            timeframe_str: Timeframe string (e.g., "1m", "5m", "1h")
            n_bars: Number of bars to fetch

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
            Returns empty DataFrame on failure.
        """
        if not self.connected:
            logger.error("MT5 not connected")
            return pd.DataFrame()

        # Ensure symbol is in Market Watch
        if not self.ensure_symbol(symbol):
            return pd.DataFrame()

        # Map timeframe string to MT5 constant
        tf = TIMEFRAME_MAP.get(timeframe_str)
        if tf is None:
            logger.error(f"Unknown timeframe: {timeframe_str}")
            return pd.DataFrame()

        # Fetch rates
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.error(f"Failed to get candles for {symbol}: {error}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)

        # Rename columns to match expected format
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        # MT5 returns tick_volume, map to volume for compatibility
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]

        # Keep only standard OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]].copy()

        logger.debug(
            f"Fetched {len(df)} bars for {symbol} ({timeframe_str}) | "
            f"Latest: {df['close'].iloc[-1]:.2f}"
        )

        return df

    def get_tick(self, symbol: str) -> dict:
        """
        Get current bid/ask tick for a symbol.

        Returns:
            Dict with bid, ask, last, time. Empty dict on failure.
        """
        if not self.connected:
            return {}

        if not self.ensure_symbol(symbol):
            return {}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "time": datetime.fromtimestamp(tick.time).isoformat(),
        }

    def get_positions(self, symbol: str = None) -> list:
        """
        Get open positions, optionally filtered by symbol.

        Returns:
            List of position dicts.
        """
        if not self.connected:
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == 0 else "SELL",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "price_current": pos.price_current,
                "profit": pos.profit,
                "swap": pos.swap,
                "magic": pos.magic,
                "comment": pos.comment,
                "time": datetime.fromtimestamp(pos.time).isoformat(),
            })

        return result

    def get_account_info(self) -> dict:
        """Get current account information."""
        if not self.connected:
            return {}

        info = mt5.account_info()
        if info is None:
            return {}

        return {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "leverage": info.leverage,
            "currency": info.currency,
        }

    def get_deals_history(self, days: int = 7) -> list:
        """
        Get trade deal history for the last N days.
        Used by trade_reviewer for self-learning.
        """
        if not self.connected:
            return []

        from datetime import timedelta
        date_from = datetime.now() - timedelta(days=days)
        date_to = datetime.now()

        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            return []

        result = []
        for deal in deals:
            if deal.magic == 240411:  # Only our bot's trades
                result.append({
                    "ticket": deal.ticket,
                    "order": deal.order,
                    "symbol": deal.symbol,
                    "type": "BUY" if deal.type == 0 else "SELL",
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit": deal.profit,
                    "swap": deal.swap,
                    "commission": deal.commission,
                    "comment": deal.comment,
                    "time": datetime.fromtimestamp(deal.time).isoformat(),
                })

        return result

    def shutdown(self):
        """Shutdown MT5 connection."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 Data Feed disconnected")
