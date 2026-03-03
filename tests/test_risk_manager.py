"""Unit tests for RiskManager."""

from __future__ import annotations

import math

import pytest

from src.risk.risk_manager import RiskManager


@pytest.fixture()
def rm() -> RiskManager:
    return RiskManager(
        {
            "max_risk_per_trade_pct": 1.0,
            "max_open_trades": 5,
            "max_daily_loss_pct": 3.0,
            "max_total_drawdown_pct": 10.0,
            "default_sl_atr_multiplier": 2.0,
            "default_tp_rr_ratio": 2.0,
            "trailing_stop": True,
            "trailing_stop_atr_multiplier": 1.5,
        }
    )


_SYMBOL_INFO = {
    "digits": 5,
    "point": 0.00001,
    "trade_contract_size": 100_000,
    "volume_min": 0.01,
    "volume_max": 500.0,
    "volume_step": 0.01,
}


class TestCanOpenTrade:
    def test_allows_when_no_limits_hit(self, rm: RiskManager) -> None:
        rm.update_account(10_000, 10_000)
        rm.set_open_trade_count(0)
        ok, reason = rm.can_open_trade(10_000)
        assert ok
        assert reason == "ok"

    def test_blocks_when_max_open_trades_reached(self, rm: RiskManager) -> None:
        rm.update_account(10_000, 10_000)
        rm.set_open_trade_count(5)
        ok, reason = rm.can_open_trade(10_000)
        assert not ok
        assert "max_open_trades" in reason

    def test_blocks_when_daily_loss_exceeded(self, rm: RiskManager) -> None:
        rm.update_account(10_000, 10_000)  # sets daily start equity
        rm.set_open_trade_count(0)
        # Simulate a 4% daily drawdown (> 3% limit)
        ok, reason = rm.can_open_trade(9_600)
        assert not ok
        assert "daily loss" in reason

    def test_blocks_when_total_drawdown_exceeded(self, rm: RiskManager) -> None:
        # Set initial balance to 10_000 and daily start equity to 9_100 so that
        # daily loss = (9100 - 8900) / 9100 ≈ 2.2% (below 3% limit) but
        # total drawdown = (10000 - 8900) / 10000 = 11% (above 10% limit).
        rm.update_account(10_000, 9_100)
        rm.set_open_trade_count(0)
        ok, reason = rm.can_open_trade(8_900)
        assert not ok
        assert "drawdown" in reason

    def test_allows_just_below_daily_limit(self, rm: RiskManager) -> None:
        rm.update_account(10_000, 10_000)
        rm.set_open_trade_count(0)
        # 2.9% daily loss — below 3% limit
        ok, _ = rm.can_open_trade(9_710)
        assert ok


class TestCalculatePositionSize:
    def test_basic_sizing(self, rm: RiskManager) -> None:
        # 1% of 10,000 = $100 risk; SL = 50 pips; pip_value_per_lot = 0.0001 × 100000 = $10
        # raw_lots = 100 / (50 × 10) = 0.2
        lots = rm.calculate_position_size(10_000, _SYMBOL_INFO, 50.0)
        assert math.isclose(lots, 0.20, rel_tol=1e-3)

    def test_zero_sl_returns_min_volume(self, rm: RiskManager) -> None:
        lots = rm.calculate_position_size(10_000, _SYMBOL_INFO, 0.0)
        assert lots == _SYMBOL_INFO["volume_min"]

    def test_result_clamped_to_min_volume(self, rm: RiskManager) -> None:
        # Very wide SL → tiny lots, but never below volume_min
        lots = rm.calculate_position_size(100, _SYMBOL_INFO, 10_000.0)
        assert lots >= _SYMBOL_INFO["volume_min"]

    def test_result_clamped_to_max_volume(self, rm: RiskManager) -> None:
        # Huge equity → capped at volume_max
        lots = rm.calculate_position_size(100_000_000, _SYMBOL_INFO, 1.0)
        assert lots <= _SYMBOL_INFO["volume_max"]

    def test_volume_step_rounding(self, rm: RiskManager) -> None:
        # Result should be a multiple of volume_step
        lots = rm.calculate_position_size(10_000, _SYMBOL_INFO, 37.0)
        step = _SYMBOL_INFO["volume_step"]
        assert math.isclose(lots % step, 0.0, abs_tol=1e-9)


class TestCalculateSlTp:
    def test_buy_sl_below_entry(self, rm: RiskManager) -> None:
        sl, tp = rm.calculate_sl_tp(1.1000, 0.0010, 0, _SYMBOL_INFO)
        assert sl < 1.1000
        assert tp > 1.1000

    def test_sell_sl_above_entry(self, rm: RiskManager) -> None:
        sl, tp = rm.calculate_sl_tp(1.1000, 0.0010, 1, _SYMBOL_INFO)
        assert sl > 1.1000
        assert tp < 1.1000

    def test_tp_respects_rr_ratio(self, rm: RiskManager) -> None:
        atr = 0.0010
        entry = 1.1000
        sl, tp = rm.calculate_sl_tp(entry, atr, 0, _SYMBOL_INFO)
        sl_dist = entry - sl
        tp_dist = tp - entry
        # tp_dist / sl_dist should equal the RR ratio (2.0)
        assert math.isclose(tp_dist / sl_dist, 2.0, rel_tol=1e-3)

    def test_custom_rr_ratio(self, rm: RiskManager) -> None:
        sl, tp = rm.calculate_sl_tp(1.1000, 0.0010, 0, _SYMBOL_INFO, custom_rr=3.0)
        sl_dist = 1.1000 - sl
        tp_dist = tp - 1.1000
        assert math.isclose(tp_dist / sl_dist, 3.0, rel_tol=1e-3)


class TestTrailingStop:
    def test_buy_trailing_below_price(self, rm: RiskManager) -> None:
        trail = rm.trailing_stop_price(1.1100, 0, 0.0010, _SYMBOL_INFO)
        assert trail < 1.1100

    def test_sell_trailing_above_price(self, rm: RiskManager) -> None:
        trail = rm.trailing_stop_price(1.1100, 1, 0.0010, _SYMBOL_INFO)
        assert trail > 1.1100


class TestSlDistanceInPips:
    def test_known_distance(self, rm: RiskManager) -> None:
        # 10 pips = 0.0010 for a 5-digit broker
        pips = rm.sl_distance_in_pips(1.1010, 1.1000, _SYMBOL_INFO)
        assert math.isclose(pips, 10.0, rel_tol=1e-3)
