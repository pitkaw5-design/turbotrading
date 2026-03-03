"""MT5 connector – thin wrapper around the MetaTrader5 Python package.

On non-Windows systems (CI, Linux servers) the real ``MetaTrader5`` package
cannot run.  This module detects that and falls back to a *mock* implementation
so that the rest of the code can be tested / developed offline.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Try to import the real MT5 package ────────────────────────────────────────
try:
    import MetaTrader5 as _mt5  # type: ignore[import]

    _MT5_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MT5_AVAILABLE = False
    _mt5 = None  # type: ignore[assignment]

# MT5 timeframe constants (numeric values match the MT5 API)
TIMEFRAMES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
    "W1": 32769,
    "MN": 49153,
}

# Order type constants
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1

_DEFAULT_MAGIC = 20240101


class MT5Connector:
    """Manages the connection to a MetaTrader 5 terminal and provides
    methods for market data retrieval and order execution.

    When ``dry_run=True`` (default) **no real orders** are sent;
    all trade operations are logged only.
    """

    def __init__(self, cfg: dict[str, Any], dry_run: bool = True) -> None:
        self._cfg = cfg.get("mt5", {})
        self._dry_run = dry_run
        self._connected = False

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise and log in to the MT5 terminal.

        Returns ``True`` on success, ``False`` otherwise.
        """
        if not _MT5_AVAILABLE:
            logger.warning("MetaTrader5 package not available – running in offline mode.")
            self._connected = False
            return False

        if not _mt5.initialize(
            login=self._cfg.get("login", 0),
            password=self._cfg.get("password", ""),
            server=self._cfg.get("server", ""),
            timeout=self._cfg.get("timeout", 60000),
        ):
            logger.error("MT5 initialize() failed: %s", _mt5.last_error())
            return False

        self._connected = True
        info = _mt5.account_info()
        logger.info(
            "Connected to MT5 | account=%s balance=%.2f currency=%s",
            info.login if info else "?",
            info.balance if info else 0.0,
            info.currency if info else "?",
        )
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 connection."""
        if _MT5_AVAILABLE and self._connected:
            _mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_info(self) -> dict[str, Any]:
        """Return key account metrics as a plain dict."""
        if not self._connected or not _MT5_AVAILABLE:
            return {"balance": 0.0, "equity": 0.0, "margin_free": 0.0, "currency": "USD"}

        info = _mt5.account_info()
        if info is None:
            return {}
        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
        }

    # ── Market data ───────────────────────────────────────────────────────────

    def get_rates(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for *symbol* / *timeframe*.

        Returns a :class:`pandas.DataFrame` with columns
        ``[open, high, low, close, tick_volume, spread, real_volume]``
        indexed by ``time`` (UTC, tz-aware).
        """
        tf_value = TIMEFRAMES.get(timeframe.upper())
        if tf_value is None:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Valid: {list(TIMEFRAMES)}")

        if not self._connected or not _MT5_AVAILABLE:
            logger.debug("MT5 not connected – returning empty rates for %s %s", symbol, timeframe)
            return pd.DataFrame()

        rates = _mt5.copy_rates_from_pos(symbol, tf_value, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning("No rates returned for %s %s: %s", symbol, timeframe, _mt5.last_error())
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        return df

    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Return point size, digits, and contract specs for *symbol*."""
        if not self._connected or not _MT5_AVAILABLE:
            return {}

        info = _mt5.symbol_info(symbol)
        if info is None:
            return {}
        return {
            "symbol": info.name,
            "digits": info.digits,
            "point": info.point,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
        }

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return a list of currently open positions (optionally filtered by symbol)."""
        if not self._connected or not _MT5_AVAILABLE:
            return []

        positions = _mt5.positions_get(symbol=symbol) if symbol else _mt5.positions_get()
        if positions is None:
            return []
        return [p._asdict() for p in positions]  # type: ignore[attr-defined]

    # ── Order execution ───────────────────────────────────────────────────────

    def open_trade(
        self,
        symbol: str,
        order_type: int,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "TurboTrading",
        magic: int = _DEFAULT_MAGIC,
    ) -> dict[str, Any]:
        """Send a market order to MT5.

        When ``dry_run=True`` the order is *simulated* and logged only.

        Parameters
        ----------
        symbol:
            Instrument name, e.g. ``"EURUSD"``.
        order_type:
            ``ORDER_TYPE_BUY`` (0) or ``ORDER_TYPE_SELL`` (1).
        volume:
            Lot size (already validated by :class:`RiskManager`).
        price:
            Execution price (``ask`` for buy, ``bid`` for sell).
        sl:
            Stop-loss price.
        tp:
            Take-profit price.
        comment:
            Order comment visible in MT5 terminal.
        magic:
            Magic number to identify agent orders.

        Returns
        -------
        dict
            Result dict with at least keys ``retcode``, ``order``, ``comment``.
        """
        action_name = "BUY" if order_type == ORDER_TYPE_BUY else "SELL"

        if self._dry_run:
            result = {
                "retcode": 10009,
                "order": 0,
                "comment": "dry_run",
                "symbol": symbol,
                "type": action_name,
                "volume": volume,
                "price": price,
                "sl": sl,
                "tp": tp,
            }
            logger.info("[DRY RUN] %s %s | vol=%.2f price=%.5f sl=%.5f tp=%.5f", action_name, symbol, volume, price, sl, tp)
            return result

        if not self._connected or not _MT5_AVAILABLE:
            raise RuntimeError("MT5 not connected – cannot send order.")

        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": _mt5.ORDER_FILLING_IOC,
        }
        result = _mt5.order_send(request)
        if result is None or result.retcode != 10009:
            logger.error("Order failed: %s | request=%s", _mt5.last_error(), request)
            return {"retcode": result.retcode if result else -1, "order": 0, "comment": str(_mt5.last_error())}

        logger.info(
            "Order executed: %s %s | ticket=%s vol=%.2f price=%.5f sl=%.5f tp=%.5f",
            action_name, symbol, result.order, volume, price, sl, tp,
        )
        return result._asdict()  # type: ignore[attr-defined]

    def close_trade(self, ticket: int, symbol: str, volume: float, order_type: int, magic: int = _DEFAULT_MAGIC) -> dict[str, Any]:
        """Close an open position by ticket.

        When ``dry_run=True`` the close is simulated and logged only.
        """
        if self._dry_run:
            logger.info("[DRY RUN] CLOSE ticket=%s %s vol=%.2f", ticket, symbol, volume)
            return {"retcode": 10009, "order": 0, "comment": "dry_run"}

        if not self._connected or not _MT5_AVAILABLE:
            raise RuntimeError("MT5 not connected – cannot close trade.")

        close_type = ORDER_TYPE_SELL if order_type == ORDER_TYPE_BUY else ORDER_TYPE_BUY
        info = _mt5.symbol_info_tick(symbol)
        price = info.bid if close_type == ORDER_TYPE_SELL else info.ask

        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": magic,
            "comment": "TurboTrading close",
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": _mt5.ORDER_FILLING_IOC,
        }
        result = _mt5.order_send(request)
        if result is None or result.retcode != 10009:
            logger.error("Close failed: %s", _mt5.last_error())
            return {"retcode": result.retcode if result else -1, "order": 0}
        logger.info("Position closed: ticket=%s", ticket)
        return result._asdict()  # type: ignore[attr-defined]
