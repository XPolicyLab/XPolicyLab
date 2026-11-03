# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guard case ownership, then delegate recovery to untouched original code.

This is an explicit operation on a stopped episode. It never resumes at initial
startup and does not interpret model failures differently from recovery.py.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
IDENTITY = (
    "training_set",
    "model",
    "reasoning_effort",
    "environment_revision",
    "test_episodes",
)


@dataclass(frozen=True)
class ResumePreparation:
    """Original recovery decision and persisted artifacts.

    Attributes:
        recovered: Whether original prepare_retry accepted recovery.
        config_path: Original runtime configuration when accepted.
        receipt_path: Audit receipt for this explicit decision.
        state: Original row including retained retry counters and history.
    """

    recovered: bool
    config_path: Path | None
    receipt_path: Path
    state: dict[str, Any]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _owned(path: Path, case: Path, *, must_exist: bool = True) -> Path:
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(case) or path == case:
        raise ValueError("Recovery path is outside its own case")
    current = path
    while current != case:
        if current.is_symlink():
            raise ValueError("Recovery records must not traverse symlinks")
        current = current.parent
    if must_exist and not path.exists():
        raise ValueError(f"Missing own-episode recovery evidence: {path.name}")
    return path


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _source_module(runtime: Path) -> Any:
    original = runtime / "recovery.py"
    pins = _json(ROOT / "provenance/source-files.json")
    expected = next(
        row["sha256"] for row in pins["files"] if row["path"] == "source/recovery.py"
    )
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
        raise ValueError("Runtime recovery.py differs from the original")
    spec = importlib.util.spec_from_file_location(
        "_original_owned_icl_recovery", original
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load original recovery module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same_identity(config: dict[str, Any], fresh: dict[str, Any]) -> None:
    if any(config.get(key) != fresh.get(key) for key in IDENTITY):
        raise ValueError("Recovery episode identity differs from this fresh case")
    if not 0 < config["episode_seconds"] <= fresh["episode_seconds"]:
        raise ValueError("Recovery budget exceeds this fresh episode")
    # These are original task/policy options; replay and device relocation do
    # not establish a different episode or permit a different action policy.
    for key in (
        "task",
        "action_mode",
        "max_repeat",
        "codex_compact_tokens",
        "review_window_steps",
        "review_max_windows",
        "review_allow_reset",
        "evaluation_independent",
        "stop_min_steps",
        "vla_horizon",
        "live_context_recovery",
    ):
        if config["options"].get(key) != fresh["options"].get(key):
            raise ValueError(f"Recovery policy differs in {key}")


def _validate_chain(
    run: Path, case: Path, fresh: dict[str, Any]
) -> list[dict[str, Any]]:
    """Require only this case's native records, preserving original replay order."""
    history = []
    seen = set()
    while True:
        _owned(run, case)
        if not run.is_relative_to(case / "runs"):
            raise ValueError("Recovery source is outside this case's original runs")
        if run in seen:
            raise ValueError("Recovery replay chain contains a cycle")
        seen.add(run)
        config_path = _owned(run / "config.json", case)
        config = _json(config_path)
        _same_identity(config, fresh)
        replay = config["options"].get("replay")
        observation = run / "observations/latest.json"
        records = list((run / "episodes").glob("*/result.json"))
        if observation.exists():
            _owned(observation, case)
            if len(records) != 1:
                raise ValueError("Recovery needs one native episode result")
            result_path = _owned(records[0], case)
            episode_path = _owned(result_path.parent / "episode.json", case)
            if _json(episode_path) != fresh["test_episodes"][0]:
                raise ValueError("Native episode identity differs from this fresh case")
            obs = _json(observation)["observation"]
            step = obs["control_step"]
            if type(step) is not int or step < 0:
                raise ValueError("Invalid native checkpoint step")
            actions_path = result_path.parent / "actions.jsonl"
            actions = []
            if actions_path.exists():
                _owned(actions_path, case)
                actions = [
                    json.loads(line) for line in actions_path.read_text().splitlines()
                ]
            history.append({"run": str(run), "checkpoint": step, "actions": actions})
        else:
            # Original recovery may retry a failed reset with no new actions.
            # Do not fabricate observations or manufacture a replay checkpoint.
            for action_file in (run / "episodes").glob("*/actions.jsonl"):
                _owned(action_file, case)
                if action_file.stat().st_size:
                    raise ValueError("Native actions exist without checkpoint evidence")
        if not replay:
            break
        record = _owned(Path(replay["source_record"]), case)
        if replay["episode_id"] != fresh["test_episodes"][0]["id"]:
            raise ValueError("Replay belongs to another episode")
        if type(replay["control_step"]) is not int or replay["control_step"] <= 0:
            raise ValueError("Invalid replay control step")
        run = record.parent.parent
    history.reverse()
    completed = 0
    for entry in history:
        for row in entry["actions"]:
            if row.get("status") != "completed":
                continue
            step = row["control_step"]
            if step > entry["checkpoint"]:
                break
            if step != completed + 1:
                raise ValueError("Native action history is not contiguous")
            completed = step
        if completed != entry["checkpoint"]:
            raise ValueError("Incomplete native action history for checkpoint")
    return [
        {"run": entry["run"], "checkpoint": entry["checkpoint"]} for entry in history
    ]


def prepare_resume(
    originalruntime_root: Path,
    instance: str,
    source_run: Path,
    state_path: Path,
    *,
    preparation_root: Path | None = None,
    experiment_root: Path | None = None,
    infrastructure_reason: str | None = None,
) -> ResumePreparation:
    """Prepare an explicit own-episode retry using original prepare_retry.

    Call only after controller and native writers have stopped and complete
    native records have been transferred into the original run directory.

    Args:
        originalruntime_root: This case's private copied original source tree.
        instance: Original task/seed identity of the fresh case.
        source_run: Stopped original run below the same case's runs directory.
        state_path: Private persistent original recovery row; retains counters.
        preparation_root: generate_fresh_config output, default case root.
        experiment_root: Private case scope containing runs, source and state.
        infrastructure_reason: Explicit infrastructure diagnosis supplied by
            lifecycle owner, using the original recovery row contract.

    Returns:
        Original decision, configuration and audit receipt. Rejection does not
        grant a new budget or create another episode.
    """
    runtime = originalruntime_root.resolve()
    case = (experiment_root or runtime.parent).resolve()
    _owned(originalruntime_root, case)
    _owned(runtime / "recovery.py", case)
    prep = (preparation_root or case).resolve()
    if prep != case:
        _owned(preparation_root or prep, case)
    receipt = _json(prep / "PREPARATION_RECEIPT.json")
    if (
        receipt.get("instance") != instance
        or not receipt.get("fresh_episode")
        or receipt.get("evaluation_state_imported") is not False
    ):
        raise ValueError("Missing authenticated fresh case identity")
    bindings = _json(ROOT / "resources/frozen-bundles.json")["task_mappings"]
    if instance not in {entry["task_instance"] for entry in bindings}:
        raise ValueError("Unknown original task instance")
    template_path = ROOT / "resources/fresh-configs" / f"{instance}.json"
    if (
        not template_path.is_file()
        or receipt.get("source_template_sha256")
        != hashlib.sha256(template_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Fresh case template identity differs from original")
    fresh = _json(prep / "import/config.json")
    template = _json(template_path)
    _same_identity(fresh, {**template, "training_set": fresh["training_set"]})
    if any(fresh.get(key) != template.get(key) for key in IDENTITY[1:]):
        raise ValueError("Fresh case identity differs from original task")
    if (
        len(fresh["test_episodes"]) != 1
        or fresh["episode_seconds"] != 86400
        or fresh["options"].get("replay")
    ):
        raise ValueError("Preparation anchor is not an original fresh episode")
    state_path = _owned(state_path, case, must_exist=False)
    config_path = _owned(runtime / "config" / f"{instance}.json", case)
    _same_identity(_json(config_path), fresh)
    source_run = _owned(source_run, case)
    summary_path = _owned(source_run / "summary.json", case)
    with _lock(state_path.with_suffix(state_path.suffix + ".lock")):
        chain = _validate_chain(source_run, case, fresh)
        summary = _json(summary_path)
        if summary.get("status") != "error":
            raise ValueError("Only an interrupted original run can be resumed")
        row = _json(state_path) if state_path.exists() else {}
        if row.get("instance", instance) != instance:
            raise ValueError("Recovery counters belong to another task")
        row["instance"] = instance
        row["run"] = str(source_run)
        row["summary"] = summary
        if infrastructure_reason is not None:
            row["infrastructure_recovery"] = {"reason": infrastructure_reason}
        module = _source_module(runtime)
        accepted = bool(module.prepare_retry(instance, row))
        _atomic(state_path, row)
        decision_path = state_path.with_suffix(".receipt.json")
        decision = {
            "instance": instance,
            "source_run": str(source_run),
            "accepted": accepted,
            "native_checkpoint_chain": chain,
            "recovery_count": row.get("recovery_count", 0),
            "recovery_block": row.get("recovery_block"),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "original_recovery_sha256": hashlib.sha256(
                (runtime / "recovery.py").read_bytes()
            ).hexdigest(),
            "remaining_seconds": _json(config_path)["episode_seconds"]
            if accepted
            else None,
            "step_offset": _json(config_path)["options"]
            .get("replay", {})
            .get("control_step", 0)
            if accepted
            else None,
        }
        _atomic(decision_path, decision)
        return ResumePreparation(
            accepted, config_path if accepted else None, decision_path, row
        )
