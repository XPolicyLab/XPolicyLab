"""Explicit binding of a capability name to an immutable version."""

from PhysicalRSI_core.contracts import Operation


class Registry:
    def __init__(self):
        self._operations: dict[str, Operation] = {}

    def register(self, operation: Operation) -> None:
        if operation.name in self._operations:
            raise ValueError(f"Already registered: {operation.name}")
        self._operations[operation.name] = operation

    def resolve(self, name: str, revision: str) -> Operation:
        operation = self._operations[name]
        if operation.revision != revision:
            raise ValueError(f"Version mismatch: {name}")
        return operation

    def list(self) -> tuple[Operation, ...]:
        return tuple(self._operations.values())
