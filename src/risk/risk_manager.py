"""Risk and capital management module.

Enforces position sizing, drawdown limits, and trade filtering before
any order is submitted to MT5.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


class RiskManager:
    """Centralised risk and money-management engine.

    Parameters
    ----------
    cfg:
        The ``risk`` sub-section of the settings dictionary.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._max_risk_pct: float = cfg.get("max_risk_per_trade_pct", 1.0)
        self._max_open_trades: int = cfg.get("max_open_trades", 5)
        self._max_daily_loss_pct: float = cfg.get("max_daily_loss_pct", 3.0)
        self._max_drawdown_pct: float = cfg.get("max_total_drawdown_pct", 10.0)
        self._sl_atr_mult: float = cfg.get("default_sl_atr_multiplier", 2.0)
        self._tp_rr: float = cfg.get("default_tp_rr_ratio", 2.0)
        self._trailing_stop: bool = cfg.get("trailing_stop", True)
        self._trailing_atr_mult: float = cfg.get("trailing_stop_atr_multiplier", 1.5)

        # Runtime state (updated each cycle by the agent)
        self._initial_balance: float = 0.0
        self._daily_start_equity: float = 0.0
        self._daily_start_date: date = date.today()
        self._open_trade_count: int = 0

    # ── State setters ─────────────────────────────────────────────────────────

    def update_account(self, balance: float, equity: float) -> None:
        """Update the risk manager with current account figures.

        Must be called at the start of each analysis cycle.
        """
        if self._initial_balance == 0.0:
            self._initial_balance = balance

        today = date.today()
        if today != self._daily_start_date:
            self._daily_start_equity = equity
            self._daily_start_date = today

        if self._daily_start_equity == 0.0:
            self._daily_start_equity = equity

    def set_open_trade_count(self, count: int) -> None:
        self._open_trade_count = count

    # ── Guard checks ──────────────────────────────────────────────────────────

    def can_open_trade(self, equity: float) -> tuple[bool, str]:
        """Return ``(True, "ok")`` if a new trade may be opened, or
        ``(False, reason)`` if risk limits prevent it.

        Parameters
        ----------
        equity:
            Current account equity.

        Returns
        -------
        tuple[bool, str]
            First element is whether opening is allowed; second is the reason.
        """
        # Max concurrent positions
        if self._open_trade_count >= self._max_open_trades:
            return False, f"max_open_trades ({self._max_open_trades}) reached"

        # Daily loss circuit-breaker
        if self._daily_start_equity > 0:
            daily_loss_pct = (self._daily_start_equity - equity) / self._daily_start_equity * 100
            if daily_loss_pct >= self._max_daily_loss_pct:
                return False, (
                    f"daily loss limit reached: {daily_loss_pct:.2f}% "
                    f">= {self._max_daily_loss_pct}%"
                )

        # Total drawdown from initial balance
        if self._initial_balance > 0:
            total_dd_pct = (self._initial_balance - equity) / self._initial_balance * 100
            if total_dd_pct >= self._max_drawdown_pct:
                return False, (
                    f"total drawdown limit reached: {total_dd_pct:.2f}% "
                    f">= {self._max_drawdown_pct}%"
                )

        return True, "ok"

    # ── Position sizing ───────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        equity: float,
        symbol_info: dict[str, Any],
        sl_pips: float,
    ) -> float:
        """Compute the lot size that risks exactly ``max_risk_per_trade_pct``
        of *equity* given a stop-loss distance in pips.

        Parameters
        ----------
        equity:
            Current account equity in account currency.
        symbol_info:
            Dict returned by ``MT5Connector.get_symbol_info()``.
            Required keys: ``point``, ``trade_contract_size``, ``volume_min``,
            ``volume_max``, ``volume_step``.
        sl_pips:
            Stop-loss distance in *pips* (not points).

        Returns
        -------
        float
            Lot size rounded to the nearest valid ``volume_step``, clamped to
            ``[volume_min, volume_max]``.
        """
        if sl_pips <= 0 or equity <= 0:
            return symbol_info.get("volume_min", 0.01)

        risk_amount = equity * (self._max_risk_pct / 100.0)

        # 1 pip = 10 × point for 5-digit brokers, point for 3-digit (JPY, etc.)
        digits = symbol_info.get("digits", 5)
        point = symbol_info.get("point", 0.00001)
        pip_value = point * (10 if digits in (5, 3) else 1)

        contract_size = symbol_info.get("trade_contract_size", 100_000)
        pip_value_per_lot = pip_value * contract_size

        if pip_value_per_lot == 0:
            return symbol_info.get("volume_min", 0.01)

        raw_lots = risk_amount / (sl_pips * pip_value_per_lot)

        # Round to valid step
        step = symbol_info.get("volume_step", 0.01)
        lots = math.floor(raw_lots / step) * step
        lots = max(symbol_info.get("volume_min", 0.01), lots)
        lots = min(symbol_info.get("volume_max", 500.0), lots)

        return round(lots, 2)

    # ── SL / TP calculation ───────────────────────────────────────────────────

    def calculate_sl_tp(
        self,
        entry_price: float,
        atr: float,
        order_type: int,
        symbol_info: dict[str, Any],
        custom_rr: float | None = None,
    ) -> tuple[float, float]:
        """Compute stop-loss and take-profit prices.

        SL = entry ± ATR × ``sl_atr_multiplier``
        TP = entry ± SL_distance × ``tp_rr_ratio``

        Parameters
        ----------
        entry_price:
            Price at which the trade will open.
        atr:
            Current ATR value (same units as price).
        order_type:
            ``0`` = BUY, ``1`` = SELL (MT5 constants).
        symbol_info:
            Dict with ``digits`` key for rounding.
        custom_rr:
            Override the configured risk/reward ratio.

        Returns
        -------
        tuple[float, float]
            ``(stop_loss_price, take_profit_price)``.
        """
        digits = symbol_info.get("digits", 5)
        rr = custom_rr if custom_rr is not None else self._tp_rr
        sl_distance = atr * self._sl_atr_mult

        if order_type == 0:  # BUY
            sl = round(entry_price - sl_distance, digits)
            tp = round(entry_price + sl_distance * rr, digits)
        else:  # SELL
            sl = round(entry_price + sl_distance, digits)
            tp = round(entry_price - sl_distance * rr, digits)

        return sl, tp

    def sl_distance_in_pips(
        self,
        entry_price: float,
        sl_price: float,
        symbol_info: dict[str, Any],
    ) -> float:
        """Convert an absolute SL distance to pips."""
        point = symbol_info.get("point", 0.00001)
        digits = symbol_info.get("digits", 5)
        pip = point * (10 if digits in (5, 3) else 1)
        return abs(entry_price - sl_price) / pip if pip > 0 else 0.0

    # ── Trailing stop ─────────────────────────────────────────────────────────

    def trailing_stop_price(
        self,
        current_price: float,
        order_type: int,
        atr: float,
        symbol_info: dict[str, Any],
    ) -> float:
        """Compute the new trailing stop price given current price and ATR."""
        digits = symbol_info.get("digits", 5)
        distance = atr * self._trailing_atr_mult
        if order_type == 0:  # BUY – trail below price
            return round(current_price - distance, digits)
        else:  # SELL – trail above price
            return round(current_price + distance, digits)
