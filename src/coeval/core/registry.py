"""Plugin registry system for CoEval components."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from coeval.datasets.base import GoldenDatasetBase
    from coeval.metrics.base import BaseMetric

logger = logging.getLogger(__name__)

_metric_registry: dict[str, type[BaseMetric]] = {}
_dataset_registry: dict[str, type[GoldenDatasetBase]] = {}


def register_dataset[T](name: str) -> Callable[[type[T]], type[T]]:
    """
    Decorator to register a dataset class.

    Supports GoldenDatasetBase classes.

    Usage:
        @register_dataset("pubmedqa")
        class PubMedQADataset(GoldenDatasetBase):
            ...
    """

    def decorator(cls: type[T]) -> type[T]:
        key = name.lower()
        if key in _dataset_registry:
            logger.warning(f"Overriding dataset registration for {key}")
        # Type cast to satisfy type checker (GoldenDatasetBase is only in TYPE_CHECKING)
        _dataset_registry[key] = cast("type[GoldenDatasetBase]", cls)  # type: ignore[assignment]
        cls._registry_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def register_metric(name: str) -> Callable[[type[BaseMetric]], type[BaseMetric]]:
    """
    Decorator to register a metric class.

    Usage:
        @register_metric("mcq_accuracy")
        class MCQAccuracyMetric(BaseMetric):
            ...
    """

    def decorator(cls: type[BaseMetric]) -> type[BaseMetric]:
        key = name.lower()
        if key in _metric_registry:
            logger.warning(f"Overriding metric registration for {key}")
        _metric_registry[key] = cls
        cls._registry_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def get_dataset(name: str, **kwargs: Any) -> Any:
    """Get a registered dataset instance (GoldenDatasetBase)."""
    key = name.lower()
    if key not in _dataset_registry:
        available = list_datasets()
        raise KeyError(
            f"No dataset registered with name={name}. Available: {available}"
        )
    return _dataset_registry[key](**kwargs)


def get_metric(name: str, **kwargs: Any) -> BaseMetric:
    """Get a registered metric instance."""
    key = name.lower()
    if key not in _metric_registry:
        available = list_metrics()
        raise KeyError(f"No metric registered with name={name}. Available: {available}")
    return _metric_registry[key](**kwargs)


def list_datasets() -> list[str]:
    """List all registered dataset names."""
    return list(_dataset_registry.keys())


def list_metrics() -> list[str]:
    """List all registered metric names."""
    return list(_metric_registry.keys())


def get_dataset_class(name: str) -> type | None:
    """Get the class (not instance) for a registered dataset."""
    return _dataset_registry.get(name.lower())


def get_metric_class(name: str) -> type | None:
    """Get the class (not instance) for a registered metric."""
    return _metric_registry.get(name.lower())


def clear_registries() -> None:
    """Clear all registries. For testing only."""
    _dataset_registry.clear()
    _metric_registry.clear()
