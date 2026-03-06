"""TurboTrading – main entry point.

Usage
-----
    # Dry-run (no real orders):
    python main.py

    # Live trading (real MT5 orders):
    python main.py --live

    # Single analysis cycle and exit:
    python main.py --once

    # Custom config:
    python main.py --config /path/to/settings.yaml

Environment variables (can also be set in a .env file):
    MT5_LOGIN          MT5 account number
    MT5_PASSWORD       MT5 account password
    MT5_SERVER         MT5 broker server name
    OPENAI_API_KEY     OpenAI API key for LLM agent
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.agent.trading_agent import TradingAgent


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="turbotrading",
        description="TurboTrading – AI forex trading agent for MetaTrader 5",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to YAML settings file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (real orders).  Default is dry-run.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single analysis cycle and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging(args.verbose)
    logger = logging.getLogger("turbotrading")

    dry_run = not args.live
    if not dry_run:
        logger.warning(
            "⚠  LIVE TRADING MODE ENABLED – real orders will be sent to MT5!"
        )

    agent = TradingAgent(config_path=args.config, dry_run=dry_run)

    if args.once:
        actions = agent.run_once()
        logger.info("Single cycle complete.  Actions taken: %d", len(actions))
        for action in actions:
            logger.info("  → %s", action)
    else:
        agent.start()


if __name__ == "__main__":
    main()
