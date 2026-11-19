"""Select local corrections without mistaking later completion for recovery."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

TASKS = ("align_blocks", "arrange_largest_number", "build_tower", "classify_objects")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_role(segment, reviewed_range, range_index):
    if segment.get("local_subgoal_only"):
        return "local_subgoal_unverified_terminal", 0.25
    witness = segment.get("handback_evidence") or {}
    if witness.get("successful_policy_continuation_without_later_agent_actions") is True:
        return "repair_then_policy_success", 1.0
    terminal = segment.get("terminal_evidence") or {}
    if terminal.get("native_success") is not True:
        raise ValueError("Positive completion requires recorded terminal evidence")
    # Only the first reviewed local operation is attached to the takeover.
    # Later FINISH operations remain auxiliary labels, not new failure events.
    if range_index == 0:
        return "takeover_local_correction", 0.75
    return "subsequent_completion", 0.20


def build_event_overlay(manifest, selection, source_root):
    segments = {s["segment_uid"]: s for s in selection["segments"]}
    decisions = {}
    events = {}
    rows = []
    active = [r for r in manifest["samples"] if r["split"] != "reserved_test"]
    scene_splits, layout_splits = defaultdict(set), defaultdict(set)
    for scene in manifest["scenes"]:
        split = "heldout" if scene["split"] == "validation" else scene["split"]
        scene_splits[scene["source_id"]].add(split)
        layout_splits[scene["layout_sha256"]].add(split)
    if any(len(v) != 1 for v in list(scene_splits.values()) + list(layout_splits.values())):
        raise ValueError("Whole source/layout leakage across splits")
    for row in active:
        sid = row["source_id"]
        if sid not in decisions:
            path = Path(source_root) / "provenance/sources" / sid / "sidecar/process/trajectory.jsonl"
            records = [json.loads(line) for line in path.open()]
            decisions[sid] = [r for r in records if r.get("kind") == "decision_applied"
                              and r.get("decision", {}).get("kind") in ("takeover", "set-goal")]
        segment = segments[row["segment_uid"]]
        if segment["action_source"] not in ("repair", "finish"):
            raise ValueError("Policy failures cannot be behavior-cloning targets")
        lo, hi = row["range_native_steps_inclusive"]
        reviewed = segment["reviewed_ranges"]
        matches = [(i, r) for i, r in enumerate(reviewed) if r["native_steps_inclusive"] == [lo, hi]]
        if len(matches) != 1:
            raise ValueError("Each window must belong to one reviewed local operation")
        range_index, rr = matches[0]
        role, quality = local_role(segment, rr, range_index)
        anchors = [d for d in decisions[sid] if d["step"] <= lo]
        if not anchors:
            raise ValueError("No causal takeover evidence")
        anchor = max(anchors, key=lambda d: d["step"])
        step = row["native_step"]
        if not lo <= step <= step + row["valid_action_steps"] - 1 <= hi:
            raise ValueError("Supervision crosses its reviewed operation boundary")
        if any(t > step for t in row["history_indices"]):
            raise ValueError("Future observation leakage")
        event_id = row.get("range_uid", f"{row['segment_uid']}:{lo}-{hi}")
        event = dict(event_id=event_id, source_id=sid, task=row["task"],
                     split="heldout" if row["split"] == "validation" else row["split"],
                     role=role, quality_weight=quality,
                     failure_decision_step=anchor["step"], failure_evidence=anchor["decision"]["evidence"],
                     repair_start=lo, repair_end_inclusive=hi, local_stage=rr.get("local_stage"),
                     onset_gap_actions=lo-anchor["step"],
                     terminal_success_verified=bool((segment.get("terminal_evidence") or {}).get("native_success")
                         or (segment.get("handback_evidence") or {}).get("successful_policy_continuation_without_later_agent_actions")),
                     executable_ee_teacher_verified=False,
                     label_contract="absolute requested-joint FK EE16; execution equivalence pending")
        events[event_id] = event
        rows.append(dict(sample_id=row["sample_id"], event_id=event_id, role=role,
                         task=row["task"], scene=sid, split=event["split"],
                         quality_weight=quality, offset_in_local_operation=step-lo,
                         onset_window=step-lo < 16,
                         local_stage=event["local_stage"]))
    summary = {}
    for task in TASKS:
        selected = [r for r in rows if r["task"] == task and r["split"] == "train"]
        summary[task] = dict(train_scenes=len({r["scene"] for r in selected}),
                             train_windows=len(selected), local_operations=len({r["event_id"] for r in selected}),
                             roles=dict(Counter(r["role"] for r in selected)))
    return dict(schema="recoverwam-local-events-v1", samples=rows, events=list(events.values()),
                summary=summary, selection_sha256=None,
                scope="Development on previously exposed source scenes; not independent benchmark evidence",
                failure_evidence_level="recorded agent observation; not an independently replay-qualified failure state",
                limitations=["No verified full-task tower recovery exists in this release.",
                             "Local completion and terminal recovery are separate labels.",
                             "Historical joint commands need EE execution parity testing.",
                             "The reserved source scenes are never sampled."])


def event_groups(rows):
    groups = {t: defaultdict(list) for t in TASKS}
    for i, row in enumerate(rows):
        if row["kind"] == "recovery" and row["split"] == "train":
            groups[row["task"]][row["event_id"]].append(i)
    if not all(groups.values()):
        raise ValueError("Missing target task; do not silently substitute a normal demonstration")
    return groups


def choose_local_window(groups, rows, rng, step):
    task = TASKS[(step - 1) % 4]
    cycle = (step - 1) // 4
    local = [k for k, ids in groups[task].items() if rows[ids[0]]["role"] != "subsequent_completion"]
    auxiliary = [k for k, ids in groups[task].items() if rows[ids[0]]["role"] == "subsequent_completion"]
    use_aux = bool(auxiliary) and cycle % 5 == 4
    keys = sorted(auxiliary if use_aux else local)
    if not keys:
        raise ValueError("No eligible local operation")
    visit = cycle // 5 if use_aux else (cycle - cycle // 5 if auxiliary else cycle)
    event = keys[visit % len(keys)]
    ids = groups[task][event]
    onset = [i for i in ids if rows[i]["onset_window"]]
    pool = onset if onset and cycle % 2 == 0 else ids
    return rng.choice(pool)
