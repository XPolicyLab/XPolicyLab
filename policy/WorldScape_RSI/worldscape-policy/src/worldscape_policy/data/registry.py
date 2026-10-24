from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, Protocol, TypeVar

from worldscape_policy.data.schema import EventSample


class EventDataset(Protocol):
    def __iter__(self) -> Iterator[EventSample]: ...


DatasetT = TypeVar("DatasetT", bound=EventDataset)
DatasetFactory = Callable[..., EventDataset]


class DatasetRegistry:
    """Minimal explicit registry for native EventSample datasets."""

    def __init__(self) -> None:
        self._factories: dict[str, DatasetFactory] = {}

    def register(
        self, name: str, factory: DatasetFactory | None = None
    ) -> DatasetFactory | Callable[[DatasetFactory], DatasetFactory]:
        normalized = _normalize_name(name)

        def add(candidate: DatasetFactory) -> DatasetFactory:
            if normalized in self._factories:
                raise ValueError(f"dataset {normalized!r} is already registered")
            self._factories[normalized] = candidate
            return candidate

        return add if factory is None else add(factory)

    def create(self, name: str, **kwargs: Any) -> EventDataset:
        normalized = _normalize_name(name)
        try:
            factory = self._factories[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"unknown dataset {name!r}; registered datasets: {available}"
            ) from exc
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def factories(self) -> Mapping[str, DatasetFactory]:
        return dict(self._factories)


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("dataset name must be non-empty")
    return normalized


DATASETS = DatasetRegistry()
