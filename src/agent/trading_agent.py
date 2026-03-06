"""AI Trading Agent – LLM-powered decision engine for MT5 forex trading.

Architecture
------------
The agent runs an analysis loop:
1. Fetch account state and open positions from MT5.
2. For each watched instrument, retrieve OHLCV data and run technical analysis.
3. Fetch the economic calendar and derive fundamental context.
4. Feed all data to the LLM (OpenAI function-calling) which decides:
   - Open a BUY / SELL trade
   - Close an existing position
   - Do nothing (NEUTRAL)
5. Validate every LLM-proposed trade through ``RiskManager`` before execution.
6. Sleep for ``cycle_interval_seconds`` and repeat.

The agent is safe to use with ``dry_run=True`` (default) – no real orders are
sent; all decisions are logged to stdout.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import openai

from ..analysis.fundamental import FundamentalAnalyzer
from ..analysis.technical import TechnicalAnalyzer
from ..connectors.mt5_connector import MT5Connector, ORDER_TYPE_BUY, ORDER_TYPE_SELL
from ..risk.risk_manager import RiskManager
from ..utils.config import load_config

logger = logging.getLogger(__name__)


# ── Tool / function schemas for OpenAI function-calling ───────────────────────

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_technical_analysis",
            "description": (
                "Returns a technical analysis summary for a given trading instrument, "
                "including price, moving averages, RSI, MACD, ATR, Bollinger Bands, "
                "Stochastic, and a composite signal (BUY/SELL/NEUTRAL)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "MT5 symbol name, e.g. EURUSD, XAUUSD, USOIL.",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe code: M1 M5 M15 M30 H1 H4 D1 W1.",
                        "default": "H1",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamental_context",
            "description": (
                "Returns upcoming and recent high-impact economic calendar events "
                "relevant to the given trading instrument."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "MT5 symbol name.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_trade",
            "description": (
                "Open a new market-order trade after all analysis confirms a setup. "
                "The risk manager will validate and may reject the order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "MT5 symbol name."},
                    "direction": {
                        "type": "string",
                        "enum": ["BUY", "SELL"],
                        "description": "Trade direction.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief justification for the trade (logged and audited).",
                    },
                },
                "required": ["symbol", "direction", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_trade",
            "description": "Close an existing open position by ticket number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket": {
                        "type": "integer",
                        "description": "MT5 position ticket.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for closing the position.",
                    },
                },
                "required": ["ticket", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_open_positions",
            "description": "Return all currently open MT5 positions with PnL and metadata.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class TradingAgent:
    """LLM-powered AI trading agent for MT5 forex / commodity trading.

    Parameters
    ----------
    config_path:
        Optional path to a YAML config file.  Defaults to
        ``config/settings.yaml`` in the project root.
    dry_run:
        When ``True`` (default) no real orders are submitted to MT5.
    """

    def __init__(
        self,
        config_path: str | None = None,
        dry_run: bool | None = None,
    ) -> None:
        self._cfg = load_config(config_path)
        agent_cfg = self._cfg.get("agent", {})

        # Honour explicit parameter, then config, then default True
        if dry_run is not None:
            self._dry_run = dry_run
        else:
            self._dry_run = agent_cfg.get("dry_run", True)

        # Sub-components
        self._mt5 = MT5Connector(self._cfg, dry_run=self._dry_run)
        self._technical = TechnicalAnalyzer(self._cfg.get("technical", {}))
        self._fundamental = FundamentalAnalyzer(self._cfg.get("fundamental", {}))
        self._risk = RiskManager(self._cfg.get("risk", {}))

        # OpenAI client
        api_key = agent_cfg.get("openai_api_key")
        self._openai = openai.OpenAI(api_key=api_key) if api_key else None
        self._model: str = agent_cfg.get("model", "gpt-4o")
        self._temperature: float = agent_cfg.get("temperature", 0.1)
        self._max_iterations: int = agent_cfg.get("max_iterations", 10)
        self._cycle_interval: int = agent_cfg.get("cycle_interval_seconds", 300)

        # Instruments
        instr = self._cfg.get("instruments", {})
        self._instruments: list[str] = (
            instr.get("forex", []) + instr.get("commodities", [])
        )
        self._primary_timeframe: str = self._cfg.get("technical", {}).get("timeframe", "H1")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect to MT5 and begin the main trading loop.

        Blocks indefinitely.  Press Ctrl+C to stop.
        """
        connected = self._mt5.connect()
        if not connected:
            logger.warning("Running in offline/dry-run mode (no MT5 connection).")

        mode = "DRY-RUN" if self._dry_run else "LIVE"
        logger.info("TurboTrading agent started | mode=%s instruments=%s", mode, self._instruments)

        try:
            while True:
                self._run_cycle()
                logger.info("Cycle complete.  Sleeping %ds …", self._cycle_interval)
                time.sleep(self._cycle_interval)
        except KeyboardInterrupt:
            logger.info("Agent stopped by user.")
        finally:
            self._mt5.disconnect()

    def run_once(self) -> list[dict[str, Any]]:
        """Run a single analysis cycle and return the list of actions taken.

        Useful for testing and one-shot execution.
        """
        return self._run_cycle()

    # ── Core cycle ────────────────────────────────────────────────────────────

    def _run_cycle(self) -> list[dict[str, Any]]:
        """Execute one full analysis → decision → execution cycle."""
        actions: list[dict[str, Any]] = []

        # 1. Account state
        account = self._mt5.get_account_info()
        equity = account.get("equity", 0.0)
        balance = account.get("balance", 0.0)
        self._risk.update_account(balance, equity)

        # 2. Open positions
        positions = self._mt5.get_open_positions()
        self._risk.set_open_trade_count(len(positions))

        # 3. Analyse each instrument via LLM
        for symbol in self._instruments:
            try:
                result = self._analyse_symbol(symbol, equity, positions)
                if result:
                    actions.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error analysing %s: %s", symbol, exc)

        return actions

    def _analyse_symbol(
        self,
        symbol: str,
        equity: float,
        positions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Run the LLM tool-calling loop for a single symbol."""
        if self._openai is None:
            return self._fallback_rule_based(symbol, equity)

        system_prompt = self._build_system_prompt(equity, positions)
        user_prompt = (
            f"Analyse {symbol} and decide whether to open a new trade, "
            f"close an existing position, or stay neutral.  "
            f"Current open positions: {json.dumps(positions, default=str)}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for iteration in range(self._max_iterations):
            response = self._openai.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                tools=_TOOLS,
                tool_choice="auto",
                messages=messages,
            )
            msg = response.choices[0].message

            # No more tool calls → final decision
            if not msg.tool_calls:
                logger.info("[%s] LLM decision: %s", symbol, msg.content)
                return {"symbol": symbol, "action": "neutral", "llm_response": msg.content}

            # Execute tool calls requested by the LLM
            messages.append(msg)  # type: ignore[arg-type]
            for tool_call in msg.tool_calls:
                tool_result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments),
                    symbol,
                    equity,
                    positions,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, default=str),
                })

            if iteration == self._max_iterations - 1:
                logger.warning("[%s] Max LLM iterations reached.", symbol)

        return None

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        default_symbol: str,
        equity: float,
        positions: list[dict[str, Any]],
    ) -> Any:
        """Dispatch an LLM tool call to the correct handler."""
        symbol = args.get("symbol", default_symbol)

        if name == "get_technical_analysis":
            tf = args.get("timeframe", self._primary_timeframe)
            lookback = self._cfg.get("technical", {}).get("lookback_bars", 500)
            df = self._mt5.get_rates(symbol, tf, lookback)
            return self._technical.analyze(df)

        elif name == "get_fundamental_context":
            return {
                "summary": self._fundamental.get_market_sentiment_summary(symbol),
                "blackout": self._fundamental.is_news_blackout(symbol),
                "upcoming_events": self._fundamental.get_upcoming_events(60),
            }

        elif name == "get_open_positions":
            return positions

        elif name == "open_trade":
            return self._handle_open_trade(
                symbol=symbol,
                direction=args.get("direction", "BUY"),
                reason=args.get("reason", ""),
                equity=equity,
            )

        elif name == "close_trade":
            return self._handle_close_trade(
                ticket=args.get("ticket", 0),
                reason=args.get("reason", ""),
                positions=positions,
            )

        return {"error": f"Unknown tool: {name}"}

    # ── Trade handlers ────────────────────────────────────────────────────────

    def _handle_open_trade(
        self,
        symbol: str,
        direction: str,
        reason: str,
        equity: float,
    ) -> dict[str, Any]:
        """Validate and execute a new trade order."""
        # Risk guard
        allowed, reason_blocked = self._risk.can_open_trade(equity)
        if not allowed:
            logger.info("[%s] Trade blocked: %s", symbol, reason_blocked)
            return {"status": "blocked", "reason": reason_blocked}

        # News blackout guard
        if self._fundamental.is_news_blackout(symbol):
            logger.info("[%s] Trade blocked: news blackout.", symbol)
            return {"status": "blocked", "reason": "news_blackout"}

        # Fetch latest price and ATR
        lookback = self._cfg.get("technical", {}).get("lookback_bars", 500)
        df = self._mt5.get_rates(symbol, self._primary_timeframe, lookback)
        analysis = self._technical.analyze(df)
        atr = analysis.get("atr")
        price = analysis.get("price")

        symbol_info = self._mt5.get_symbol_info(symbol)
        if not symbol_info and not self._dry_run:
            return {"status": "error", "reason": "symbol_info_unavailable"}

        # Use defaults when offline / dry-run
        if not symbol_info:
            symbol_info = {
                "digits": 5,
                "point": 0.00001,
                "trade_contract_size": 100_000,
                "volume_min": 0.01,
                "volume_max": 500.0,
                "volume_step": 0.01,
                "bid": price or 1.0,
                "ask": (price or 1.0) + 0.00010,
            }

        order_type = ORDER_TYPE_BUY if direction.upper() == "BUY" else ORDER_TYPE_SELL
        entry_price = symbol_info.get("ask" if order_type == ORDER_TYPE_BUY else "bid", price or 1.0)

        if atr is None or atr == 0:
            atr = entry_price * 0.001  # 0.1% fallback ATR

        sl, tp = self._risk.calculate_sl_tp(entry_price, atr, order_type, symbol_info)
        sl_pips = self._risk.sl_distance_in_pips(entry_price, sl, symbol_info)
        volume = self._risk.calculate_position_size(equity or 10_000, symbol_info, sl_pips)

        logger.info(
            "Opening %s %s | price=%.5f sl=%.5f tp=%.5f vol=%.2f | reason: %s",
            direction, symbol, entry_price, sl, tp, volume, reason,
        )

        result = self._mt5.open_trade(
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            price=entry_price,
            sl=sl,
            tp=tp,
        )
        # Keep the intra-cycle risk counter in sync so that subsequent symbols
        # in the same cycle correctly see the updated position count.
        self._risk.increment_open_trade_count()
        return {"status": "executed", "order": result, "reason": reason}

    def _handle_close_trade(
        self,
        ticket: int,
        reason: str,
        positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Close an existing position by ticket."""
        pos = next((p for p in positions if p.get("ticket") == ticket), None)
        if pos is None:
            return {"status": "error", "reason": f"ticket {ticket} not found in open positions"}

        symbol = pos.get("symbol", "")
        volume = pos.get("volume", 0.01)
        order_type = pos.get("type", 0)

        logger.info("Closing position ticket=%s %s vol=%.2f | reason: %s", ticket, symbol, volume, reason)
        result = self._mt5.close_trade(ticket, symbol, volume, order_type)
        return {"status": "closed", "ticket": ticket, "result": result, "reason": reason}

    # ── Fallback (no OpenAI key) ───────────────────────────────────────────────

    def _fallback_rule_based(
        self,
        symbol: str,
        equity: float,
    ) -> dict[str, Any] | None:
        """Simple rule-based fallback when OpenAI API is not configured.

        Uses the technical composite signal directly without LLM reasoning.
        """
        lookback = self._cfg.get("technical", {}).get("lookback_bars", 500)
        df = self._mt5.get_rates(symbol, self._primary_timeframe, lookback)
        if df.empty:
            return None

        analysis = self._technical.analyze(df)
        signal = analysis.get("signal", "NEUTRAL")
        strength = analysis.get("signal_strength", 0.0)

        if signal == "NEUTRAL":
            logger.info("[%s] Fallback: NEUTRAL (strength=%.3f)", symbol, strength)
            return None

        reason = (
            f"Rule-based signal: {signal} | RSI={analysis.get('rsi')} "
            f"MACD_hist={analysis.get('macd_hist')} strength={strength}"
        )
        logger.info("[%s] Fallback: %s | %s", symbol, signal, reason)

        return self._handle_open_trade(
            symbol=symbol,
            direction=signal,
            reason=reason,
            equity=equity,
        )

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        equity: float,
        positions: list[dict[str, Any]],
    ) -> str:
        return (
            "You are TurboTrading, an expert AI forex and commodity trading agent. "
            "Your goal is to identify high-probability trading opportunities on major "
            "forex pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD), "
            "gold (XAUUSD), and crude oil (USOIL) on a MetaTrader 5 platform.\n\n"
            "Rules:\n"
            "1. ALWAYS call get_technical_analysis before deciding on any trade.\n"
            "2. ALWAYS call get_fundamental_context to check for news blackouts.\n"
            "3. Do NOT open a trade if fundamental context indicates a blackout.\n"
            "4. Only open trades when technical signals are strongly aligned "
            "   (BUY/SELL signal strength > 0.3 and multiple indicator confirmation).\n"
            "5. Risk management is enforced automatically – do not override it.\n"
            "6. Prefer closing losing positions before opening new ones.\n"
            f"7. Current account equity: {equity:.2f}.\n"
            f"8. Open positions: {len(positions)}.\n"
        )
