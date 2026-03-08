"""Persistence helpers for saved strategy payloads."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autonomous_trading_researcher.research.strategy_generator import GeneratedStrategy
from autonomous_trading_researcher.strategies.base import BaseStrategy
from autonomous_trading_researcher.strategies.breakout import BreakoutStrategy
from autonomous_trading_researcher.strategies.mean_reversion import MeanReversionStrategy
from autonomous_trading_researcher.strategies.momentum import MomentumStrategy

BUILTIN_STRATEGY_CLASSES = {
    MomentumStrategy.name: MomentumStrategy,
    MeanReversionStrategy.name: MeanReversionStrategy,
    BreakoutStrategy.name: BreakoutStrategy,
}


def _generated_factory(
    parameters: dict[str, float | int | str],
) -> Callable[[dict[str, float | int | str]], BaseStrategy]:
    """Build a generated strategy factory."""

    return lambda overrides, parameters=parameters: GeneratedStrategy(
        **({**parameters, **overrides})
    )


def _builtin_factory(
    strategy_class: type[BaseStrategy],
    parameters: dict[str, float | int | str],
) -> Callable[[dict[str, float | int | str]], BaseStrategy]:
    """Build a factory for a persisted built-in strategy."""

    return lambda overrides, parameters=parameters: strategy_class(
        **({**parameters, **overrides})
    )


def generated_strategy_directory(path: str | Path | None = None) -> Path:
    """Return the directory used to persist top strategies."""

    if path is not None:
        target = Path(path)
    else:
        target = Path(__file__).resolve().parent
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_strategy_payload(
    payload: dict[str, Any],
    directory: str | Path | None = None,
) -> Path:
    """Persist one strategy payload as JSON."""

    strategy_dir = generated_strategy_directory(directory)
    strategy_id = str(payload.get("strategy_id", payload["strategy_name"]))
    target = strategy_dir / f"{strategy_id}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_saved_strategy_factories(
    directory: str | Path | None = None,
) -> dict[str, Callable[[dict[str, float | int | str]], BaseStrategy]]:
    """Load persisted strategies into instantiation factories."""

    strategy_dir = generated_strategy_directory(directory)
    factories: dict[str, Callable[[dict[str, float | int | str]], BaseStrategy]] = {}
    for file_path in strategy_dir.glob("*.json"):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        parameters = payload["parameters"]
        strategy_name = str(payload["strategy_name"])
        strategy_id = str(payload.get("strategy_id", strategy_name))
        if payload.get("family") == "generated":
            factories[strategy_id] = _generated_factory(parameters)
            continue
        strategy_class = BUILTIN_STRATEGY_CLASSES.get(strategy_name)
        if strategy_class is not None:
            factories[strategy_id] = _builtin_factory(strategy_class, parameters)
    return factories
