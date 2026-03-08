"""Strategy registry and factory helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from autonomous_trading_researcher.research.strategy_generator import GeneratedStrategy
from autonomous_trading_researcher.strategies.base import BaseStrategy
from autonomous_trading_researcher.strategies.breakout import BreakoutStrategy
from autonomous_trading_researcher.strategies.generated.loader import load_saved_strategy_factories
from autonomous_trading_researcher.strategies.mean_reversion import MeanReversionStrategy
from autonomous_trading_researcher.strategies.momentum import MomentumStrategy

type StrategyFactory = Callable[[dict[str, float | int | str]], BaseStrategy]

BUILTIN_STRATEGIES: dict[str, StrategyFactory] = {
    MomentumStrategy.name: lambda parameters: MomentumStrategy(**parameters),
    MeanReversionStrategy.name: lambda parameters: MeanReversionStrategy(**parameters),
    BreakoutStrategy.name: lambda parameters: BreakoutStrategy(**parameters),
    GeneratedStrategy.name: lambda parameters: GeneratedStrategy(**parameters),
}
STRATEGY_REGISTRY: dict[str, StrategyFactory] = dict(BUILTIN_STRATEGIES)


def refresh_generated_strategies(directory: str | Path | None = None) -> None:
    """Load persisted strategies from disk into the registry."""

    STRATEGY_REGISTRY.update(load_saved_strategy_factories(directory))


def get_strategy(name: str, parameters: dict[str, float | int | str]) -> BaseStrategy:
    """Instantiate a strategy by name."""

    if name not in STRATEGY_REGISTRY:
        refresh_generated_strategies()

    if name not in STRATEGY_REGISTRY and isinstance(parameters.get("template"), str):
        return GeneratedStrategy(**parameters)

    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}")

    strategy_factory = STRATEGY_REGISTRY[name]
    return strategy_factory(parameters)


def list_registered_strategies() -> list[str]:
    """Return the registered strategy names."""

    refresh_generated_strategies()
    return sorted(STRATEGY_REGISTRY.keys())
