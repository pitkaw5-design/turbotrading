"""Unit tests for TradingAgent (no MT5 / OpenAI required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agent.trading_agent import TradingAgent


def _make_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0005, n))
    high = close + 0.001
    low = close - 0.001
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": np.ones(n) * 500,
        }
    )


@pytest.fixture()
def agent(tmp_path) -> TradingAgent:
    """Return a dry-run agent with a minimal in-memory config."""
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text(
        """
instruments:
  forex: [EURUSD]
  commodities: []
mt5:
  login: 0
  password: ""
  server: ""
technical:
  timeframe: H1
  lookback_bars: 300
  sma_fast: 20
  sma_slow: 50
  ema_trend: 200
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  atr_period: 14
  bb_period: 20
  bb_std: 2.0
  stoch_k: 5
  stoch_d: 3
  stoch_smooth: 3
fundamental:
  high_impact_only: true
  news_lookback_hours: 24
risk:
  max_risk_per_trade_pct: 1.0
  max_open_trades: 5
  max_daily_loss_pct: 3.0
  max_total_drawdown_pct: 10.0
  default_sl_atr_multiplier: 2.0
  default_tp_rr_ratio: 2.0
  trailing_stop: true
  trailing_stop_atr_multiplier: 1.5
agent:
  model: gpt-4o
  temperature: 0.1
  max_iterations: 3
  cycle_interval_seconds: 60
  dry_run: true
"""
    )
    return TradingAgent(config_path=str(cfg_file), dry_run=True)


class TestTradingAgentInit:
    def test_dry_run_default(self, agent: TradingAgent) -> None:
        assert agent._dry_run is True

    def test_instruments_loaded(self, agent: TradingAgent) -> None:
        assert "EURUSD" in agent._instruments

    def test_openai_none_when_no_key(self, agent: TradingAgent) -> None:
        # No OPENAI_API_KEY in env → _openai should be None
        assert agent._openai is None


class TestFallbackRuleBased:
    def test_returns_none_on_empty_rates(self, agent: TradingAgent) -> None:
        with patch.object(agent._mt5, "get_rates", return_value=pd.DataFrame()):
            result = agent._fallback_rule_based("EURUSD", 10_000)
        assert result is None

    def test_opens_trade_on_strong_buy_signal(self, agent: TradingAgent) -> None:
        df = _make_df()
        # Force a BUY signal by patching the technical analyzer
        with (
            patch.object(agent._mt5, "get_rates", return_value=df),
            patch.object(
                agent._technical,
                "analyze",
                return_value={
                    "signal": "BUY",
                    "signal_strength": 0.8,
                    "atr": 0.0020,
                    "price": 1.1050,
                },
            ),
            patch.object(agent._fundamental, "is_news_blackout", return_value=False),
            patch.object(agent._mt5, "get_symbol_info", return_value={}),
        ):
            result = agent._fallback_rule_based("EURUSD", 10_000)

        assert result is not None
        assert result.get("status") == "executed"

    def test_neutral_signal_returns_none(self, agent: TradingAgent) -> None:
        df = _make_df()
        with (
            patch.object(agent._mt5, "get_rates", return_value=df),
            patch.object(
                agent._technical,
                "analyze",
                return_value={"signal": "NEUTRAL", "signal_strength": 0.0},
            ),
        ):
            result = agent._fallback_rule_based("EURUSD", 10_000)
        assert result is None


class TestHandleOpenTrade:
    def test_blocked_by_risk_manager(self, agent: TradingAgent) -> None:
        agent._risk.set_open_trade_count(5)  # max reached
        agent._risk.update_account(10_000, 10_000)

        result = agent._handle_open_trade("EURUSD", "BUY", "test", 10_000)
        assert result["status"] == "blocked"

    def test_blocked_by_news_blackout(self, agent: TradingAgent) -> None:
        agent._risk.set_open_trade_count(0)
        agent._risk.update_account(10_000, 10_000)
        with patch.object(agent._fundamental, "is_news_blackout", return_value=True):
            result = agent._handle_open_trade("EURUSD", "BUY", "test", 10_000)
        assert result["status"] == "blocked"
        assert result["reason"] == "news_blackout"

    def test_dry_run_executes_without_mt5(self, agent: TradingAgent) -> None:
        agent._risk.set_open_trade_count(0)
        agent._risk.update_account(10_000, 10_000)
        df = _make_df()
        with (
            patch.object(agent._fundamental, "is_news_blackout", return_value=False),
            patch.object(agent._mt5, "get_rates", return_value=df),
            patch.object(
                agent._technical,
                "analyze",
                return_value={
                    "signal": "BUY",
                    "signal_strength": 0.7,
                    "atr": 0.0015,
                    "price": 1.1020,
                },
            ),
            patch.object(agent._mt5, "get_symbol_info", return_value={}),
            patch.object(
                agent._mt5, "open_trade", wraps=agent._mt5.open_trade
            ) as mock_open,
        ):
            result = agent._handle_open_trade("EURUSD", "BUY", "unit test", 10_000)

        assert result["status"] == "executed"
        assert result["order"]["comment"] == "dry_run"
        # Verify open_trade was called once (dry-run path, no real MT5 call)
        mock_open.assert_called_once()


class TestHandleCloseTrade:
    def test_returns_error_for_unknown_ticket(self, agent: TradingAgent) -> None:
        result = agent._handle_close_trade(9999, "test close", [])
        assert result["status"] == "error"

    def test_closes_known_position(self, agent: TradingAgent) -> None:
        positions = [{"ticket": 1, "symbol": "EURUSD", "volume": 0.01, "type": 0}]
        result = agent._handle_close_trade(1, "take profit hit", positions)
        assert result["status"] == "closed"
        assert result["ticket"] == 1


class TestRunOnce:
    def test_run_once_returns_list(self, agent: TradingAgent) -> None:
        with (
            patch.object(agent._mt5, "connect", return_value=False),
            patch.object(agent._mt5, "get_account_info", return_value={"equity": 10_000, "balance": 10_000}),
            patch.object(agent._mt5, "get_open_positions", return_value=[]),
            patch.object(agent._mt5, "get_rates", return_value=pd.DataFrame()),
        ):
            actions = agent.run_once()
        assert isinstance(actions, list)


class TestIntraCycleTradeLimit:
    def test_open_trade_increments_risk_count(self, agent: TradingAgent) -> None:
        """After a successful open_trade, the RiskManager's count must be incremented
        so that subsequent trades in the same cycle respect max_open_trades."""
        agent._risk.update_account(10_000, 10_000)
        agent._risk.set_open_trade_count(0)
        df = _make_df()

        with (
            patch.object(agent._fundamental, "is_news_blackout", return_value=False),
            patch.object(agent._mt5, "get_rates", return_value=df),
            patch.object(
                agent._technical,
                "analyze",
                return_value={"signal": "BUY", "signal_strength": 0.7, "atr": 0.0015, "price": 1.1020},
            ),
            patch.object(agent._mt5, "get_symbol_info", return_value={}),
        ):
            agent._handle_open_trade("EURUSD", "BUY", "test", 10_000)

        # Count must have been incremented from 0 to 1
        assert agent._risk._open_trade_count == 1

    def test_second_trade_blocked_after_max_reached(self, agent: TradingAgent) -> None:
        """With max_open_trades=1, the second call to _handle_open_trade in the
        same cycle must be blocked after the first trade increments the count."""
        # Override max to 1 for this test
        agent._risk._max_open_trades = 1
        agent._risk.update_account(10_000, 10_000)
        agent._risk.set_open_trade_count(0)
        df = _make_df()

        analysis_patch = {
            "signal": "BUY", "signal_strength": 0.7, "atr": 0.0015, "price": 1.1020
        }
        with (
            patch.object(agent._fundamental, "is_news_blackout", return_value=False),
            patch.object(agent._mt5, "get_rates", return_value=df),
            patch.object(agent._technical, "analyze", return_value=analysis_patch),
            patch.object(agent._mt5, "get_symbol_info", return_value={}),
        ):
            result1 = agent._handle_open_trade("EURUSD", "BUY", "first", 10_000)
            result2 = agent._handle_open_trade("GBPUSD", "BUY", "second", 10_000)

        assert result1["status"] == "executed"
        assert result2["status"] == "blocked"
        assert "max_open_trades" in result2["reason"]
