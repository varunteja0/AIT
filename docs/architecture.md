# Architecture

## System layout

The repository is organized as an autonomous quantitative research laboratory with explicit subsystem boundaries:

- `data`: CCXT ingestion, parquet storage, and historical dataset construction.
- `features`: classical indicators plus market microstructure features.
- `strategies`: built-in strategies, generated strategy persistence, and registry loading.
- `backtesting`: event-driven simulation, vectorized simulation, and parallel batch execution.
- `research`: strategy generation, large-scale discovery, Bayesian optimization, genetic evolution, and experiment tracking.
- `risk`: portfolio limits and trading halts.
- `execution`: paper trading and `ccxt` execution services.
- `monitoring`: state snapshots, event logs, and dashboard-facing status persistence.
- `ui`: FastAPI dashboard and Plotly visualizations.
- `orchestration`: the autonomous loop coordinating the full pipeline.

## Data pipeline

1. Exchange trades and order book snapshots are collected through `MarketDataCollector`.
2. Raw events are stored in parquet partitions in the data lake.
3. The parquet layer validates required schema, normalizes timestamps to UTC, deduplicates overlapping incremental batches, and supports filtered historical reads for incremental research jobs.
3. `HistoricalDatasetBuilder` aggregates trades into OHLCV bars and enriches them with:
   - buy and sell volume
   - trade count
   - top-of-book prices
   - top-of-book depth
4. Bar datasets are aligned to the configured timeframe so downstream feature and backtest logic sees a consistent time grid.
5. `FeaturePipeline` computes both traditional indicators and microstructure signals.

## Feature engine

The feature engine combines classic quantitative features with market microstructure:

- returns and log returns
- rolling volatility
- fast and slow moving averages
- momentum
- liquidity score
- order book imbalance
- microprice
- trade intensity
- volume delta
- spread
- order flow imbalance

Features are assembled through an ordered registry, which keeps the pipeline extensible without hard-coding every new signal into one method. This keeps generated strategy research and traditional strategies on the same feature surface.

## Strategy discovery engine

Discovery now spans three families:

1. Traditional indicator strategies from the built-in registry.
2. Generated feature-rule strategies built from thousands of random feature combinations.
3. Microstructure-aware generated strategies emphasizing order flow, imbalance, and queue dynamics.

Discovery evaluates large populations in parallel with the vectorized backtester, validates shortlisted winners with the event-driven backtester, runs walk-forward out-of-sample checks on the best candidates, persists experiments to SQLite, and saves the best deployable strategies under `strategies/generated/`.

## Backtesting and validation

Two complementary simulation paths are used:

- the vectorized engine for large population sweeps and parallel parameter evaluation
- the event-driven engine for sequential validation with next-bar execution semantics

The event-driven validator now lags exposures by one bar to avoid lookahead bias. Shortlisted strategies also pass through walk-forward validation using expanding train/test windows before final ranking.

## Genetic evolution engine

`GeneticStrategyEvolutionEngine` evolves generated strategies using:

- seeded population initialization
- Sharpe-driven fitness ranking with multi-metric scoring
- tournament selection
- feature-threshold mutation
- parameter crossover
- elitist survival across generations

The result is a higher-quality generated strategy population that can be re-ranked and deployed automatically.

## Execution system

Execution remains risk-gated:

- paper trading is the default deployment mode
- live execution uses `ccxt`
- the risk manager enforces projected position size, projected exposure, daily loss, and drawdown limits before order submission
- the execution layer updates local portfolio state after fills

## UI dashboard

The FastAPI dashboard exposes:

- `/api/strategies`
- `/api/top_strategies`
- `/api/system_status`
- `/api/metrics`

The UI renders:

- system status
- strategy test counts
- top-ranked strategies
- equity curve
- drawdown
- feature importance
- feature correlation heatmap
- optimization scatter plots
- recent event-log entries
