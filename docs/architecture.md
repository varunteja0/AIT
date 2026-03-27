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
2. The collector supports historical pagination, per-symbol `since` checkpoints, checkpoint persistence, and automatic resume from prior parquet state.
3. Raw events are stored in parquet partitions by dataset, exchange, symbol, and date in the data lake.
4. The parquet layer validates required schema, normalizes timestamps to UTC, deduplicates overlapping incremental batches, and supports filtered historical reads for incremental research jobs.
5. `HistoricalDatasetBuilder` aggregates trades into OHLCV bars and enriches them with:
   - buy and sell volume
   - trade count
   - top-of-book prices
   - top-of-book depth
6. Multi-resolution datasets are built for `1s`, `5s`, `1m`, and `5m` bars with UTC-normalized, deduplicated, gap-filled indexes.
7. `FeaturePipeline` computes both traditional indicators and microstructure signals.

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
- VWAP distance
- order book slope
- liquidity pressure

Features are assembled through an ordered registry, which keeps the pipeline extensible without hard-coding every new signal into one method. The pipeline now also performs multi-timeframe alignment so the anchor timeframe carries unsuffixed features while higher and lower timeframe features are exposed as suffixed columns such as `momentum_1s`, `volatility_1m`, and `vwap_distance_5m`.

## Strategy discovery engine

Discovery now spans three families:

1. Traditional indicator strategies from the built-in registry.
2. Generated feature-rule strategies built from thousands of random feature combinations.
3. Microstructure-aware generated strategies emphasizing order flow, imbalance, and queue dynamics.

Discovery evaluates large populations in parallel with the vectorized backtester, validates shortlisted winners with the event-driven backtester, runs walk-forward out-of-sample checks on the best candidates, applies statistical acceptance thresholds, persists experiments to SQLite, persists Optuna studies to SQLite, and saves the best deployable strategies under `strategies/generated/`.

## Backtesting and validation

Two complementary simulation paths are used:

- the vectorized engine for large population sweeps and parallel parameter evaluation
- the event-driven engine for sequential validation with next-bar execution semantics

The event-driven validator now lags exposures by one bar to avoid lookahead bias. Shortlisted strategies also pass through walk-forward validation using expanding train/test windows before final ranking.

Backtesting also respects strategy-level position rules such as `holding_period`, `stop_loss`, and `take_profit`, which are used by both generated strategies and Bayesian search.

## Genetic evolution engine

`GeneticStrategyEvolutionEngine` evolves generated strategies using:

- seeded population initialization
- Sharpe-driven fitness ranking with multi-metric scoring
- tournament selection
- feature-threshold mutation
- parameter crossover
- elitist survival across generations

The result is a higher-quality generated strategy population that can be re-ranked and deployed automatically.

## Statistical validation

Deployable strategies pass an explicit statistical screen:

- Sharpe ratio floor
- Sortino ratio floor
- profit factor floor
- maximum drawdown ceiling
- t-statistic of mean alpha

This prevents the deployment path from treating walk-forward completion alone as sufficient evidence.

## Execution system

Execution remains risk-gated:

- paper trading is the default deployment mode
- live execution uses `ccxt`
- the paper broker simulates latency, slippage, and transaction costs
- the risk manager enforces projected position size, projected exposure, daily loss, drawdown, volatility targeting, and Kelly-bounded sizing before order submission
- an ensemble engine selects the top validated strategies and combines their signals with weighted voting
- the execution layer updates local portfolio state after fills

## Autonomous loop

The orchestration layer now supports both single-cycle and continuous operation. Each cycle performs:

1. data collection and checkpoint advancement
2. multi-resolution dataset construction
3. multi-timeframe feature generation
4. large-scale discovery and Optuna refinement
5. walk-forward and statistical validation
6. ensemble selection and paper or live deployment
7. monitoring snapshot emission and event logging

When new cycles replace prior ensemble members, the old strategies are retired automatically and logged as such.

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
- Sharpe ratio
- win rate
- trade count
- feature importance
- feature correlation heatmap
- optimization scatter plots
- recent event-log entries
