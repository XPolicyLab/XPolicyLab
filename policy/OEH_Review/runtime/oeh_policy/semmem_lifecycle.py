"""Episode closure is lifecycle bookkeeping, never task-completion evidence."""
import json
import os
from pathlib import Path
import time


def close_episode(runtime, reason):
    if not runtime.cfg.get("semmem"):
        return None
    directory = Path(runtime.output) / "semmem"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "episode-closeout.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("episode_id") != runtime.episode_id:
            raise ValueError("SemMem closeout episode identity mismatch")
        return saved
    tree = getattr(runtime, "task_tree", None)
    state = tree.state if tree else {}
    root = state.get("nodes", {}).get(state.get("root_id"), {})
    record = {"schema": "oeh.semmem.episode-closeout.v1", "episode_id": runtime.episode_id,
        "lifecycle_status": "closed", "reason": reason, "closed_at": time.time(),
        "native_step": runtime.gate.step, "tree_revision": state.get("revision"),
        "root_status_at_close": root.get("status"),
        "achievement_assessment": "reported_achieved" if root.get("status") == "achieved" else "unconfirmed",
        "assessment_source": "stored_primary_llm_plan" if tree else None,
        "pending_node_ids": [list(q["node_ids"]) for q in state.get("queues", [])],
        "completion_evidence_added": False}
    pending = path.with_suffix(".tmp")
    try:
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    runtime.emit("semmem_episode_closed", record)
    return record
