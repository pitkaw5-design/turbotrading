"""Configuration loader for TurboTrading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file if present (overrides existing env vars only when not already set)
load_dotenv(override=False)

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "settings.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration, merging environment variable overrides.

    Parameters
    ----------
    path:
        Path to the YAML settings file.  Defaults to ``config/settings.yaml``
        relative to the project root.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    with config_path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    # ── MT5 credentials from environment ──────────────────────────────────
    mt5_login = os.getenv("MT5_LOGIN")
    if mt5_login:
        cfg.setdefault("mt5", {})["login"] = int(mt5_login)

    mt5_password = os.getenv("MT5_PASSWORD")
    if mt5_password:
        cfg.setdefault("mt5", {})["password"] = mt5_password

    mt5_server = os.getenv("MT5_SERVER")
    if mt5_server:
        cfg.setdefault("mt5", {})["server"] = mt5_server

    # ── OpenAI API key ─────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        cfg.setdefault("agent", {})["openai_api_key"] = openai_key

    return cfg
