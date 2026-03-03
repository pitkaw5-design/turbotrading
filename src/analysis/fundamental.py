"""Fundamental data analysis module.

Fetches the economic calendar (ForexFactory JSON feed) and provides a
structured view of upcoming / recent high-impact events that may affect
open positions or new trade decisions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Currency to instrument mapping used for relevance filtering
_CURRENCY_INSTRUMENTS: dict[str, list[str]] = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "XAUUSD", "USOIL"],
    "EUR": ["EURUSD"],
    "GBP": ["GBPUSD"],
    "JPY": ["USDJPY"],
    "AUD": ["AUDUSD"],
    "CHF": ["USDCHF"],
    "CAD": ["USDCAD"],
    "XAU": ["XAUUSD"],
}


class FundamentalAnalyzer:
    """Fetches and interprets economic calendar events.

    Parameters
    ----------
    cfg:
        The ``fundamental`` sub-section of the settings dictionary.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._calendar_url: str = cfg.get(
            "calendar_url",
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        )
        self._high_impact_only: bool = cfg.get("high_impact_only", True)
        self._news_lookback_hours: int = cfg.get("news_lookback_hours", 24)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "TurboTrading/1.0"})

    # ── Public API ────────────────────────────────────────────────────────────

    def get_events(self) -> list[dict[str, Any]]:
        """Fetch and return this week's economic events.

        Returns
        -------
        list[dict]
            List of event dicts with keys:
            ``title, country, date, impact, forecast, previous``.
            Empty list on network error.
        """
        try:
            response = self._session.get(self._calendar_url, timeout=15)
            response.raise_for_status()
            events: list[dict[str, Any]] = response.json()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch economic calendar (network error): %s", exc)
            return []
        except ValueError as exc:
            logger.warning("Failed to parse economic calendar response: %s", exc)
            return []

        if self._high_impact_only:
            events = [e for e in events if e.get("impact", "").lower() == "high"]

        return events

    def get_upcoming_events(self, minutes_ahead: int = 60) -> list[dict[str, Any]]:
        """Return high-impact events scheduled within *minutes_ahead* minutes.

        Parameters
        ----------
        minutes_ahead:
            Look-ahead window in minutes.

        Returns
        -------
        list[dict]
            Filtered list of imminent events.
        """
        now_utc = datetime.now(tz=timezone.utc)
        events = self.get_events()
        upcoming = []
        for event in events:
            event_time = self._parse_event_time(event.get("date", ""))
            if event_time is None:
                continue
            delta = (event_time - now_utc).total_seconds() / 60
            if 0 <= delta <= minutes_ahead:
                event["minutes_until"] = round(delta, 1)
                upcoming.append(event)
        return upcoming

    def is_news_blackout(
        self,
        symbol: str,
        blackout_minutes_before: int = 30,
        blackout_minutes_after: int = 15,
    ) -> bool:
        """Return ``True`` if a high-impact news event is imminent or just
        occurred for currencies related to *symbol*.

        During a news blackout the agent should **not** open new trades.
        """
        now_utc = datetime.now(tz=timezone.utc)
        events = self.get_events()
        relevant_currencies = self._symbol_to_currencies(symbol)

        for event in events:
            if event.get("country", "").upper() not in relevant_currencies:
                continue
            event_time = self._parse_event_time(event.get("date", ""))
            if event_time is None:
                continue
            delta_minutes = (event_time - now_utc).total_seconds() / 60
            if -blackout_minutes_after <= delta_minutes <= blackout_minutes_before:
                logger.info(
                    "News blackout for %s: '%s' in %.0f min",
                    symbol,
                    event.get("title", ""),
                    delta_minutes,
                )
                return True
        return False

    def get_market_sentiment_summary(self, symbol: str) -> str:
        """Return a short text summary of recent / upcoming fundamental events
        that are relevant to *symbol*.

        This text is injected into the LLM prompt.
        """
        events = self.get_events()
        relevant_currencies = self._symbol_to_currencies(symbol)
        now_utc = datetime.now(tz=timezone.utc)

        relevant: list[str] = []
        for event in events:
            if event.get("country", "").upper() not in relevant_currencies:
                continue
            event_time = self._parse_event_time(event.get("date", ""))
            if event_time is None:
                continue
            delta_hours = (event_time - now_utc).total_seconds() / 3600
            if abs(delta_hours) <= self._news_lookback_hours:
                direction = "upcoming" if delta_hours > 0 else "recent"
                relevant.append(
                    f"[{direction}] {event.get('country','?')} – {event.get('title','?')} "
                    f"(impact={event.get('impact','?')}, "
                    f"forecast={event.get('forecast','?')}, "
                    f"previous={event.get('previous','?')})"
                )

        if not relevant:
            return "No significant fundamental events in the current window."
        return "\n".join(relevant)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_event_time(date_str: str) -> datetime | None:
        """Parse ISO 8601 or ForexFactory date strings."""
        if not date_str:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%m-%d-%YT%H:%M:%S"):
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return None

    @staticmethod
    def _symbol_to_currencies(symbol: str) -> list[str]:
        """Extract the two currency codes embedded in a forex symbol.

        Handles ``EURUSD`` → ``["EUR", "USD"]``,
        ``XAUUSD`` → ``["XAU", "USD"]``, ``USOIL`` → ``["USD"]``.
        """
        sym = symbol.upper().replace(".", "")
        if len(sym) == 6:
            return [sym[:3], sym[3:]]
        # Commodity or non-standard – just return USD as the dominant currency
        return ["USD"]
