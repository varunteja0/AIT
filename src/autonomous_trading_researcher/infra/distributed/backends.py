"""Distributed execution backends."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Protocol, Sequence, TypeVar

T = TypeVar("T")


class ExecutionBackend(Protocol):
    """Minimal backend protocol for distributed execution."""

    def map(self, func: Callable[..., T], items: Iterable[Sequence[object]]) -> list[T]: ...


class LocalExecutionBackend:
    """Local process-based execution backend."""

    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max(1, max_workers)

    def map(self, func: Callable[..., T], items: Iterable[Sequence[object]]) -> list[T]:
        payload = list(items)
        if not payload:
            return []
        if self.max_workers <= 1 or len(payload) == 1:
            return [func(*args) for args in payload]
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(func, *args) for args in payload]
            return [future.result() for future in futures]


class RayExecutionBackend:
    """Ray-based execution backend."""

    def __init__(self, address: str | None = None, namespace: str = "atr") -> None:
        import ray  # type: ignore

        if not ray.is_initialized():
            ray.init(address=address, namespace=namespace, ignore_reinit_error=True)
        self._ray = ray

    def map(self, func: Callable[..., T], items: Iterable[Sequence[object]]) -> list[T]:
        payload = list(items)
        if not payload:
            return []
        ray = self._ray

        @ray.remote
        def _runner(args: Sequence[object]) -> T:  # type: ignore[misc]
            return func(*args)

        futures = [_runner.remote(args) for args in payload]
        return list(ray.get(futures))


def build_backend(
    backend: str,
    *,
    max_workers: int = 1,
    ray_address: str | None = None,
    ray_namespace: str = "atr",
) -> ExecutionBackend:
    """Factory for execution backends."""

    backend_key = backend.lower()
    if backend_key == "ray":
        try:
            return RayExecutionBackend(address=ray_address, namespace=ray_namespace)
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ray backend requested but ray is not installed; install with 'pip install .[distributed]'"
            ) from exc
    return LocalExecutionBackend(max_workers=max_workers)
