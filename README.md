<p align="center">
  <img src="Logo_AIT.png">
</p>

# Autonomous AI Trading Researcher

Autonomous AI Trading Researcher is a modular quantitative trading research platform built to discover, validate, rank, and safely deploy systematic trading strategies.

## What is included

- Market data ingestion through exchange APIs with parquet storage.
- Feature engineering for returns, volatility, momentum, moving averages, order book imbalance, and liquidity metrics.
- Strategy framework with momentum, mean reversion, and breakout implementations.
- Event-driven and vectorized backtesting with transaction cost and slippage modeling.
- Automated strategy discovery using grid search, Bayesian optimization, and a genetic algorithm.
- Risk controls for position size, exposure, daily loss, and drawdown.
- Execution adapters backed by `ccxt`.
- Monitoring and an autonomous research loop that can rank and deploy the best candidate.
- Dataset versioning and feature-set manifests for reproducible research runs.
- Regime detection, portfolio allocation, and a persistent research knowledge graph.
- Distributed execution backends for scaling batch backtests.

## Repository layout

```text
config/      Example configuration files
docs/        Architecture notes
src/         Python implementation
tests/       Unit tests
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
atr research --config config/default.yaml
```

The FastAPI dashboard serves both HTML and JSON APIs. Start it with:

```bash
atr dashboard --config config/default.yaml --host 0.0.0.0 --port 8000
```

## Workflow coverage

The implementation follows the required build sequence:

1. Architecture planning
2. Data infrastructure
3. Feature engineering
4. Strategy research
5. Backtesting
6. AI strategy discovery
7. Risk management
8. Execution
9. Monitoring
10. Autonomous research loop

## Notes

- The project defaults to paper-trading safe behavior.
- Live execution requires valid exchange credentials and explicit enablement in the configuration.
- The data collector uses resilient polling via `ccxt` to ingest trades and order book snapshots.
