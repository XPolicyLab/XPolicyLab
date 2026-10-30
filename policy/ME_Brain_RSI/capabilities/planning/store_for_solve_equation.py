"""Immutable memory snapshots composable with tools and executable skills."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from PhysicalRSI_core.contracts import Contract, Operation
from PhysicalRSI_core.infra.storage import atomic_json, digest, locked, read_json


@dataclass(frozen=True)
class Snapshot:
    root: Path
    revision: str

    def read(self) -> dict:
        if len(self.revision) != 64 or any(
            c not in "0123456789abcdef" for c in self.revision
        ):
            raise ValueError("Invalid snapshot revision")
        value = read_json(self.root / (self.revision + ".json"))
        if digest(value) != self.revision:
            raise ValueError("Memory snapshot changed")
        return value

    def reader(self, key: str, output: Contract) -> Operation:
        """Expose memory as a typed source in any skill composition."""

        def retrieve(_, context):
            return deepcopy(self.read()[key])

        return Operation(
            f"memory.{key}", self.revision, Contract("unit"), output, retrieve
        )


class MemoryStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def snapshot(self, entries: dict) -> Snapshot:
        if not isinstance(entries, dict):
            raise ValueError("Memory entries must be a JSON object")
        # Freeze through JSON serialization before publishing. No implicit active pointer.
        value = deepcopy(entries)
        revision = digest(value)
        with locked(self.root / ".write.lock"):
            path = self.root / (revision + ".json")
            if path.exists():
                if digest(read_json(path)) != revision:
                    raise ValueError("Existing memory snapshot is corrupt")
            else:
                atomic_json(path, value)
        return Snapshot(self.root, revision)
