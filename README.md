# turbotrading
AI agent for profitable trading on the forex market on the MT5 platform,
incorporating fundamental and technical data as well as risk and capital
management on major currency pairs, gold, and oil.

## Features

- **MetaTrader 5 integration** — connects to any MT5-compatible broker; reads OHLCV data and submits market orders
- **Technical analysis** — SMA, EMA, RSI, MACD, ATR, Bollinger Bands, Stochastic Oscillator; composite BUY/SELL/NEUTRAL signal
- **Fundamental analysis** — high-impact economic calendar events (ForexFactory feed); news-blackout guard prevents trading during volatile releases
- **Risk & capital management**:
  - Per-trade risk (% of equity)
  - ATR-based stop-loss and take-profit
  - Configurable risk/reward ratio
  - ATR-based trailing stop
  - Max open positions limit
  - Daily drawdown circuit-breaker
  - Total equity drawdown hard stop
- **AI decision engine** — LLM-powered agent (OpenAI GPT-4o via function-calling) that analyses technical + fundamental data and decides whether to open, close, or skip trades
- **Rule-based fallback** — works without an OpenAI API key using the composite technical signal directly
- **Dry-run mode (default)** — logs all decisions and trade parameters without sending any real orders to MT5
- **Instruments**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, XAUUSD (Gold), USOIL (Crude Oil)

## Project structure

```
turbotrading/
├── config/
│   └── settings.yaml          # Default configuration
├── src/
│   ├── agent/
│   │   └── trading_agent.py   # AI trading agent (LLM + fallback)
│   ├── analysis/
│   │   ├── technical.py       # Technical indicators & composite signal
│   │   └── fundamental.py     # Economic calendar / news fetcher
│   ├── connectors/
│   │   └── mt5_connector.py   # MetaTrader 5 wrapper
│   ├── risk/
│   │   └── risk_manager.py    # Position sizing & risk controls
│   └── utils/
│       └── config.py          # YAML config loader + env overrides
├── tests/                     # Unit tests (pytest)
├── main.py                    # CLI entry point
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/pitkaw5-design/turbotrading.git
cd turbotrading

# Install Python dependencies
pip install -r requirements.txt

# On Windows, also install MetaTrader5:
pip install MetaTrader5
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```env
MT5_LOGIN=123456
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Demo
OPENAI_API_KEY=sk-...
```

All settings can also be customised in `config/settings.yaml`.

## Usage

```bash
# Dry-run (default) — logs signals, no real orders:
python main.py

# Single analysis cycle and exit:
python main.py --once

# Live trading (sends real orders to MT5):
python main.py --live

# Custom config file:
python main.py --config /path/to/settings.yaml

# Verbose logging:
python main.py --verbose
```

## Running tests

```bash
pytest tests/ -v
```

## Risk warning

> **This software is provided for educational and research purposes only.**
> Trading forex, gold, and oil involves substantial risk of loss.
> Past performance of any trading system does not guarantee future results.
> Always test thoroughly on a demo account before deploying with real funds.
