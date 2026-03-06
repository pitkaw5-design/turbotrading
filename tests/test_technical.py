"""Unit tests for TechnicalAnalyzer."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.technical import TechnicalAnalyzer


def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0005, n))
    high = close + rng.uniform(0.0001, 0.001, n)
    low = close - rng.uniform(0.0001, 0.001, n)
    open_ = close + rng.normal(0, 0.0002, n)
    volume = rng.integers(100, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": volume}
    )


@pytest.fixture()
def analyzer() -> TechnicalAnalyzer:
    cfg = {
        "sma_fast": 20,
        "sma_slow": 50,
        "ema_trend": 200,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "stoch_k": 5,
        "stoch_d": 3,
        "stoch_smooth": 3,
    }
    return TechnicalAnalyzer(cfg)


class TestTechnicalAnalyzer:
    def test_analyze_returns_signal_key(self, analyzer: TechnicalAnalyzer) -> None:
        df = _make_df()
        result = analyzer.analyze(df)
        assert "signal" in result
        assert result["signal"] in {"BUY", "SELL", "NEUTRAL"}

    def test_analyze_insufficient_data(self, analyzer: TechnicalAnalyzer) -> None:
        df = _make_df(n=10)
        result = analyzer.analyze(df)
        assert result["signal"] == "NEUTRAL"
        assert result.get("reason") == "insufficient_data"

    def test_analyze_empty_df(self, analyzer: TechnicalAnalyzer) -> None:
        result = analyzer.analyze(pd.DataFrame())
        assert result["signal"] == "NEUTRAL"

    def test_rsi_range(self, analyzer: TechnicalAnalyzer) -> None:
        df = _make_df()
        result = analyzer.analyze(df)
        rsi = result.get("rsi")
        if rsi is not None:
            assert 0 <= rsi <= 100

    def test_bollinger_bands_ordering(self, analyzer: TechnicalAnalyzer) -> None:
        df = _make_df()
        result = analyzer.analyze(df)
        upper = result.get("bb_upper")
        mid = result.get("bb_mid")
        lower = result.get("bb_lower")
        if upper is not None and mid is not None and lower is not None:
            assert upper >= mid >= lower

    def test_atr_positive(self, analyzer: TechnicalAnalyzer) -> None:
        df = _make_df()
        result = analyzer.analyze(df)
        atr = result.get("atr")
        if atr is not None:
            assert atr > 0

    def test_stochastic_range(self, analyzer: TechnicalAnalyzer) -> None:
        df = _make_df()
        result = analyzer.analyze(df)
        k = result.get("stoch_k")
        d = result.get("stoch_d")
        if k is not None:
            assert 0 <= k <= 100
        if d is not None:
            assert 0 <= d <= 100

    def test_signal_strength_magnitude(self, analyzer: TechnicalAnalyzer) -> None:
        df = _make_df()
        result = analyzer.analyze(df)
        strength = result.get("signal_strength", 0.0)
        assert -1.0 <= strength <= 1.0

    def test_sma_fast_above_slow_buy_contribution(self, analyzer: TechnicalAnalyzer) -> None:
        """When SMA_fast > SMA_slow, the trend vote should be bullish."""
        rng = np.random.default_rng(0)
        # Create a clearly uptrending series
        close = np.linspace(1.0, 1.5, 300) + rng.normal(0, 0.001, 300)
        df = pd.DataFrame({
            "open": close,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "tick_volume": np.ones(300) * 1000,
        })
        result = analyzer.analyze(df)
        # In a persistent uptrend the composite signal should be BUY or strength > 0
        assert result["signal_strength"] > 0

    def test_sma_internal(self, analyzer: TechnicalAnalyzer) -> None:
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert math.isclose(TechnicalAnalyzer._sma(close, 3), 4.0)

    def test_sma_returns_none_for_insufficient_data(self, analyzer: TechnicalAnalyzer) -> None:
        """_sma must return None (not NaN) when the series is shorter than the period."""
        close = pd.Series([1.0, 2.0, 3.0])
        result = TechnicalAnalyzer._sma(close, 50)
        assert result is None

    def test_sma_none_does_not_corrupt_composite_signal(self, analyzer: TechnicalAnalyzer) -> None:
        """When sma_slow period exceeds available bars, the signal must not be
        silently corrupted by a NaN vote.  The absent SMA should be excluded
        from voting rather than counted as a bearish -1.
        """
        cfg_wide = dict(analyzer._cfg)
        cfg_wide["sma_slow"] = 200  # wider than our 60-bar dataframe
        wide_analyzer = TechnicalAnalyzer(cfg_wide)
        rng = np.random.default_rng(7)
        close = np.linspace(1.0, 1.5, 60) + rng.normal(0, 0.0005, 60)
        df = pd.DataFrame({
            "open": close, "high": close + 0.001, "low": close - 0.001,
            "close": close, "tick_volume": np.ones(60) * 500,
        })
        result = wide_analyzer.analyze(df)
        # sma_slow must be None (not NaN) so it is correctly excluded from votes
        assert result["sma_slow"] is None
        # signal_strength must be a valid finite float – not NaN-corrupted
        assert not math.isnan(result["signal_strength"])
        # The corrected code (sma_slow excluded) must produce a >= signal_strength
        # than the buggy code (sma_slow=NaN counted as a false -1 bearish vote).
        # With the fix, one incorrect -1 vote is removed, so strength can only increase.
        assert result["signal_strength"] >= -0.2  # not falsely biased toward SELL

    def test_ema_returns_none_when_insufficient(self, analyzer: TechnicalAnalyzer) -> None:
        close = pd.Series([1.0, 2.0])
        assert TechnicalAnalyzer._ema(close, 5) is None

    def test_rsi_all_gains_returns_100(self, analyzer: TechnicalAnalyzer) -> None:
        """RSI must equal 100 when every bar in the window is a gain (no losses)."""
        close = pd.Series(list(range(1, 30)))  # strictly increasing
        rsi = TechnicalAnalyzer._rsi(close, 14)
        assert rsi is not None
        assert math.isclose(rsi, 100.0, rel_tol=1e-6)

    def test_rsi_all_losses_returns_0(self, analyzer: TechnicalAnalyzer) -> None:
        """RSI must equal 0 when every bar in the window is a loss (no gains)."""
        close = pd.Series(list(range(30, 1, -1)))  # strictly decreasing
        rsi = TechnicalAnalyzer._rsi(close, 14)
        assert rsi is not None
        assert math.isclose(rsi, 0.0, abs_tol=1e-6)

    def test_rsi_no_movement_returns_50(self, analyzer: TechnicalAnalyzer) -> None:
        """RSI must equal 50 when there is no price movement at all."""
        close = pd.Series([1.0] * 30)
        rsi = TechnicalAnalyzer._rsi(close, 14)
        assert rsi is not None
        assert math.isclose(rsi, 50.0, rel_tol=1e-6)

    def test_atr_static(self, analyzer: TechnicalAnalyzer) -> None:
        n = 30
        high = pd.Series([1.01] * n)
        low = pd.Series([0.99] * n)
        close = pd.Series([1.00] * n)
        atr = TechnicalAnalyzer._atr(high, low, close, 14)
        assert atr is not None
        # TR = high - low = 0.02 for each bar → ATR ≈ 0.02
        assert math.isclose(atr, 0.02, rel_tol=1e-4)
