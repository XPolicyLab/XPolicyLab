"""Shared execution contracts; core has no dependency on concrete capabilities."""

from dataclasses import dataclass, field
from threading import Event
from time import monotonic
from typing import Any, Callable


@dataclass(frozen=True)
class Contract:
    """Exact semantic type. Unit/frame/embodiment conversions must be explicit."""

    name: str
    unit: str = ""
    frame: str = ""
    embodiment: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("A named contract is required")


class Cancelled(RuntimeError):
    """Cooperative cancellation, including deadline expiry."""


@dataclass(frozen=True)
class Context:
    episode: str
    cancelled: Event = field(default_factory=Event)
    deadline: float | None = None
    execution: Callable | None = field(default=None, compare=False, repr=False)
    harness_revision: str = ""

    def __post_init__(self):
        from PhysicalRSI_core.infra.storage import identifier

        identifier(self.episode)

    def check(self) -> None:
        if self.cancelled.is_set() or (
            self.deadline is not None and monotonic() >= self.deadline
        ):
            raise Cancelled("Execution cancelled or deadline exceeded")


@dataclass(frozen=True)
class Operation:
    """A versioned executable endpoint shared by memory, tools and skills.

    `revision` binds the implementation and its dependencies by the provider.
    `effects` name exclusive physical/state resources, not success conditions.
    The caller owns completion validation; returning a value is not task success.
    """

    name: str
    revision: str
    input: Contract
    output: Contract
    call: Callable[[Any, Context], Any] = field(compare=False, repr=False)
    effects: frozenset[str] = frozenset()
    children: tuple = ()

    def __post_init__(self):
        if not self.name or not self.revision or not callable(self.call):
            raise ValueError("Operation requires a name, revision and callable")
        object.__setattr__(self, "effects", frozenset(self.effects))

    def __call__(self, value: Any, context: Context) -> Any:
        if context.execution is not None:
            return context.execution(self, value, context)
        context.check()
        result = self.call(value, context)
        context.check()
        return result
