"""Helpers for generating strategy parameter candidates."""

from __future__ import annotations

import itertools
import random

type ParameterValue = float | int | str
type ParameterSpace = dict[str, list[ParameterValue]]


def iter_parameter_grid(
    parameter_space: ParameterSpace,
    limit: int | None = None,
) -> list[dict[str, ParameterValue]]:
    """Enumerate a discrete parameter grid with an optional cap."""

    if not parameter_space:
        return [{}]
    keys = list(parameter_space.keys())
    combinations = itertools.product(*(parameter_space[key] for key in keys))
    parameters: list[dict[str, ParameterValue]] = []
    for index, values in enumerate(combinations):
        if limit is not None and index >= limit:
            break
        parameters.append(dict(zip(keys, values, strict=True)))
    return parameters


def sample_parameters(
    parameter_space: ParameterSpace,
    rng: random.Random,
) -> dict[str, ParameterValue]:
    """Randomly sample one candidate from a discrete parameter space."""

    return {name: rng.choice(values) for name, values in parameter_space.items()}
