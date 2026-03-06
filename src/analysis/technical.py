"""Technical analysis module.

Computes a standardised set of indicators on a price DataFrame and returns
a structured summary used by the AI agent for decision making.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta  # type: ignore[import]

    _TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TA_AVAILABLE = False

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Compute technical indicators from OHLCV price data.

    Parameters
    ----------
    cfg:
        The ``technical`` sub-section of the settings dictionary.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all configured indicators on *df* and return a summary dict.

        Parameters
        ----------
        df:
            OHLCV DataFrame with columns ``open, high, low, close, tick_volume``.
            Must contain at least 200 rows for all indicators to be valid.

        Returns
        -------
        dict
            Flat dict of scalar latest values plus a ``signal`` key
            (``"BUY"``, ``"SELL"``, or ``"NEUTRAL"``).
        """
        if df.empty or len(df) < 50:
            return {"signal": "NEUTRAL", "reason": "insufficient_data"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        cfg = self._cfg
        result: dict[str, Any] = {}

        # ── Moving averages ───────────────────────────────────────────────
        sma_fast = self._sma(close, cfg.get("sma_fast", 20))
        sma_slow = self._sma(close, cfg.get("sma_slow", 50))
        ema_trend = self._ema(close, cfg.get("ema_trend", 200))

        result["sma_fast"] = round(sma_fast, 5) if sma_fast is not None else None
        result["sma_slow"] = round(sma_slow, 5) if sma_slow is not None else None
        result["ema_trend"] = round(ema_trend, 5) if ema_trend is not None else None
        result["price"] = round(float(close.iloc[-1]), 5)

        # ── RSI ───────────────────────────────────────────────────────────
        rsi = self._rsi(close, cfg.get("rsi_period", 14))
        result["rsi"] = round(rsi, 2) if rsi is not None else None

        # ── MACD ──────────────────────────────────────────────────────────
        macd_line, macd_signal, macd_hist = self._macd(
            close,
            cfg.get("macd_fast", 12),
            cfg.get("macd_slow", 26),
            cfg.get("macd_signal", 9),
        )
        result["macd_line"] = round(macd_line, 6) if macd_line is not None else None
        result["macd_signal"] = round(macd_signal, 6) if macd_signal is not None else None
        result["macd_hist"] = round(macd_hist, 6) if macd_hist is not None else None

        # ── ATR ───────────────────────────────────────────────────────────
        atr = self._atr(high, low, close, cfg.get("atr_period", 14))
        result["atr"] = round(atr, 6) if atr is not None else None

        # ── Bollinger Bands ───────────────────────────────────────────────
        bb_upper, bb_mid, bb_lower = self._bollinger(
            close, cfg.get("bb_period", 20), cfg.get("bb_std", 2.0)
        )
        result["bb_upper"] = round(bb_upper, 5) if bb_upper is not None else None
        result["bb_mid"] = round(bb_mid, 5) if bb_mid is not None else None
        result["bb_lower"] = round(bb_lower, 5) if bb_lower is not None else None

        # ── Stochastic ────────────────────────────────────────────────────
        stoch_k, stoch_d = self._stochastic(
            high, low, close,
            cfg.get("stoch_k", 5),
            cfg.get("stoch_d", 3),
            cfg.get("stoch_smooth", 3),
        )
        result["stoch_k"] = round(stoch_k, 2) if stoch_k is not None else None
        result["stoch_d"] = round(stoch_d, 2) if stoch_d is not None else None

        # ── Composite signal ──────────────────────────────────────────────
        result["signal"], result["signal_strength"] = self._composite_signal(result, cfg)

        return result

    # ── Signal logic ──────────────────────────────────────────────────────────

    def _composite_signal(
        self,
        r: dict[str, Any],
        cfg: dict[str, Any],
    ) -> tuple[str, float]:
        """Derive a directional signal from the computed indicators.

        Uses a simple vote-based approach:
        - Trend alignment (price vs EMAs / SMAs)
        - Momentum (RSI, MACD histogram, Stochastic)
        - Mean-reversion (Bollinger Band position)

        Returns a tuple ``(signal, strength)`` where *strength* is a float
        in ``[-1.0, 1.0]`` (positive = bullish, negative = bearish).
        """
        votes: list[float] = []

        price = r.get("price")
        sma_fast = r.get("sma_fast")
        sma_slow = r.get("sma_slow")
        ema_trend = r.get("ema_trend")

        # ── Trend votes ───────────────────────────────────────────────────
        if price is not None and sma_fast is not None:
            votes.append(1.0 if price > sma_fast else -1.0)
        if sma_fast is not None and sma_slow is not None:
            votes.append(1.0 if sma_fast > sma_slow else -1.0)
        if price is not None and ema_trend is not None:
            votes.append(0.5 if price > ema_trend else -0.5)

        # ── RSI votes ─────────────────────────────────────────────────────
        rsi = r.get("rsi")
        rsi_ob = cfg.get("rsi_overbought", 70)
        rsi_os = cfg.get("rsi_oversold", 30)
        if rsi is not None:
            if rsi < rsi_os:
                votes.append(1.0)   # oversold → bullish
            elif rsi > rsi_ob:
                votes.append(-1.0)  # overbought → bearish
            else:
                votes.append(0.0)

        # ── MACD votes ────────────────────────────────────────────────────
        macd_hist = r.get("macd_hist")
        if macd_hist is not None:
            votes.append(1.0 if macd_hist > 0 else -1.0)

        # ── Stochastic votes ──────────────────────────────────────────────
        stoch_k = r.get("stoch_k")
        stoch_d = r.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            if stoch_k < 20 and stoch_d < 20:
                votes.append(1.0)
            elif stoch_k > 80 and stoch_d > 80:
                votes.append(-1.0)
            else:
                votes.append(0.0)

        if not votes:
            return "NEUTRAL", 0.0

        strength = float(np.mean(votes))
        if strength > 0.2:
            return "BUY", round(strength, 3)
        elif strength < -0.2:
            return "SELL", round(strength, 3)
        else:
            return "NEUTRAL", round(strength, 3)

    # ── Indicator helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _sma(series: pd.Series, period: int) -> float | None:
        if len(series) < period:
            return None
        val = series.rolling(period).mean().iloc[-1]
        return float(val) if not np.isnan(val) else None

    @staticmethod
    def _ema(series: pd.Series, period: int) -> float | None:
        if len(series) < period:
            return None
        return float(series.ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float | None:
        if len(series) < period + 1:
            return None
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # When loss == 0, the division produces NaN.  Correct values:
        #   gain > 0 and loss == 0  →  RSI = 100 (pure uptrend, no losses)
        #   gain == 0 and loss == 0 →  RSI = 50  (no price movement)
        zero_loss = loss == 0
        rsi = rsi.where(~zero_loss, other=100.0)
        rsi = rsi.where(~(zero_loss & (gain == 0)), other=50.0)
        val = rsi.iloc[-1]
        return float(val) if not np.isnan(val) else None

    @staticmethod
    def _macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float | None, float | None, float | None]:
        if len(series) < slow + signal:
            return None, None, None
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - macd_signal
        return float(macd_line.iloc[-1]), float(macd_signal.iloc[-1]), float(hist.iloc[-1])

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float | None:
        if len(close) < period + 1:
            return None
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else None

    @staticmethod
    def _bollinger(
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> tuple[float | None, float | None, float | None]:
        if len(series) < period:
            return None, None, None
        mid = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        return float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1])

    @staticmethod
    def _stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 5,
        d_period: int = 3,
        smooth: int = 3,
    ) -> tuple[float | None, float | None]:
        if len(close) < k_period + d_period + smooth:
            return None, None
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        k = raw_k.rolling(smooth).mean()
        d = k.rolling(d_period).mean()
        k_val = k.iloc[-1]
        d_val = d.iloc[-1]
        return (
            float(k_val) if not np.isnan(k_val) else None,
            float(d_val) if not np.isnan(d_val) else None,
        )
