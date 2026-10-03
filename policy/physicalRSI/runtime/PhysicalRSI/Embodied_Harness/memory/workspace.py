"""Run-scoped drafts and immutable candidates; publication belongs to Self-Harness.

Adapted from the upstream inbox/published-memory separation. Drafting never
updates an active memory pointer and conflicts are preserved as separate drafts.
"""

import uuid

from PhysicalRSI_core.contracts import Contract, Operation
from PhysicalRSI_core.infra.storage import atomic_json, identifier, read_json

from .store import MemoryStore


class MemoryWorkspace:
    def __init__(self, store: MemoryStore, parent, run_id):
        self.store = store
        self.parent = parent
        self.root = store.root / "inbox" / identifier(run_id)

    def propose(self, changes: dict, *, evidence: dict) -> dict:
        if not isinstance(changes, dict) or not evidence:
            raise ValueError("Memory changes and development evidence are required")
        draft_id = uuid.uuid4().hex
        draft = dict(
            id=draft_id,
            parent=self.parent.revision,
            changes=changes,
            evidence=evidence,
            state="candidate",
        )
        atomic_json(self.root / (draft_id + ".json"), draft)
        return draft

    def materialize(self, draft_id):
        draft = read_json(self.root / (identifier(draft_id) + ".json"))
        if draft["parent"] != self.parent.revision:
            raise ValueError("Draft belongs to another memory parent")
        entries = self.parent.read()
        entries.update(draft["changes"])
        return self.store.snapshot(entries)

    def tools(self):
        """Ordinary Operations: reusable in a skill or exported through MCP."""
        return (
            Operation(
                "memory.read",
                self.parent.revision,
                Contract("unit"),
                Contract("memory-entries"),
                lambda _, ctx: self.parent.read(),
            ),
            Operation(
                "memory.propose",
                self.parent.revision,
                Contract("memory-proposal"),
                Contract("memory-draft"),
                lambda value, ctx: self.propose(
                    value["changes"], evidence=value["evidence"]
                ),
                frozenset({str(self.root)}),
            ),
        )
