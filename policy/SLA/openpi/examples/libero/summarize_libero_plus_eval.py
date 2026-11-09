#!/usr/bin/env python3
"""Summarize LIBERO-plus eval logs into 7-dimension scores and overall score.

Example:
  python scripts/summarize_libero_plus_eval.py \
    --logs logs/vlibero1_libero_spatial_plus_output.log \
           logs/vlibero1_libero_object_plus_output.log \
           logs/vlibero1_libero_goal_plus_output.log \
           logs/vlibero1_libero_10_plus_output.log \
    --task-classification third_party/libero_plus/libero/libero/benchmark/task_classification.json \
    --json-out logs/vlibero1_libero_plus_summary.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]

CATEGORY_TO_SCORE_KEY = {
    "Camera Viewpoints": "camera",
    "Robot Initial States": "robot",
    "Language Instructions": "language",
    "Light Conditions": "light",
    "Background Textures": "background",
    "Sensor Noise": "noise",
    "Objects Layout": "layout",
}


@dataclass
class EpisodeRecord:
    suite: Optional[str]
    task_text: str
    success: bool
    source_log: str
    line_no: int


def _normalize_text(text: str) -> str:
    text = text.strip().lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_scene_prefix(name: str) -> str:
    # Example:
    #   KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1
    # -> turn_on_the_stove_and_put_the_moka_pot_on_it_table_1
    return re.sub(r"^[A-Za-z]+(?:_[A-Za-z]+)*_SCENE\d+_", "", name)


def _read_task_classification(path: pathlib.Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    lookup: Dict[str, Dict[str, str]] = {}
    ordered_categories: Dict[str, List[str]] = {}
    for suite, entries in raw.items():
        suite_map: Dict[str, str] = {}
        cat_order: List[str] = []
        for item in entries:
            raw_name = item["name"]
            cat_order.append(item["category"])
            variants = [raw_name, _strip_scene_prefix(raw_name)]
            for v in variants:
                norm = _normalize_text(v)
                if norm and norm not in suite_map:
                    suite_map[norm] = item["category"]
        lookup[suite] = suite_map
        ordered_categories[suite] = cat_order
    return lookup, ordered_categories


def _parse_log(log_path: pathlib.Path, suite_override: Optional[str]) -> List[EpisodeRecord]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    suite: Optional[str] = suite_override
    current_task: Optional[str] = None
    wait_task_text = False
    out: List[EpisodeRecord] = []

    for i, line in enumerate(lines):
        m_suite = re.search(r"Task suite:\s*([a-zA-Z0-9_]+)", line)
        if m_suite and suite is None:
            suite = m_suite.group(1).strip()

        m_task = re.search(r"Task:\s*(.*)$", line)
        if m_task:
            maybe_task = m_task.group(1).strip()
            if maybe_task:
                current_task = maybe_task
                wait_task_text = False
            else:
                wait_task_text = True
            continue

        if wait_task_text:
            stripped = line.strip()
            if stripped and not stripped.startswith("INFO:root:"):
                current_task = stripped
                wait_task_text = False
                continue

        m_success = re.search(r"Success:\s*(True|False)", line)
        if m_success and current_task:
            out.append(
                EpisodeRecord(
                    suite=suite,
                    task_text=current_task,
                    success=(m_success.group(1) == "True"),
                    source_log=str(log_path),
                    line_no=i + 1,
                )
            )

    return out


def _find_task_classification_default() -> Optional[pathlib.Path]:
    candidates = [
        pathlib.Path("LIBERO-plus/libero/libero/benchmark/task_classification.json"),
        pathlib.Path("../LIBERO-plus/libero/libero/benchmark/task_classification.json"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize LIBERO-plus eval logs.")
    p.add_argument("--logs", nargs="+", required=True, help="One or more eval log paths.")
    p.add_argument(
        "--task-classification",
        default=None,
        help="Path to task_classification.json. If omitted, try common local paths.",
    )
    p.add_argument(
        "--suite-override",
        default=None,
        help="Force suite name for logs that do not contain 'Task suite: ...'.",
    )
    p.add_argument(
        "--overall",
        choices=["weighted", "mean7d"],
        default="weighted",
        help="Overall type: 'weighted' (official task-count weighting) or 'mean7d'.",
    )
    p.add_argument("--json-out", default=None, help="Optional path to save summary JSON.")
    return p


def _rate(successes: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return 100.0 * successes / total


def _print_summary(
    *,
    episodes: List[EpisodeRecord],
    matched: List[Tuple[EpisodeRecord, str]],
    unmatched: List[EpisodeRecord],
    use_weighted_overall: bool,
) -> Dict:
    by_cat_total = defaultdict(int)
    by_cat_success = defaultdict(int)
    by_suite_total = defaultdict(int)
    by_suite_success = defaultdict(int)

    for rec, cat in matched:
        by_cat_total[cat] += 1
        by_cat_success[cat] += int(rec.success)
        suite_key = rec.suite or "unknown"
        by_suite_total[suite_key] += 1
        by_suite_success[suite_key] += int(rec.success)

    dim_scores = {}
    print("Dimension Scores (%):")
    for cat in CATEGORY_ORDER:
        succ = by_cat_success[cat]
        tot = by_cat_total[cat]
        score = _rate(succ, tot)
        key = CATEGORY_TO_SCORE_KEY[cat]
        dim_scores[key] = score
        score_text = "null" if score is None else f"{score:.4f}"
        print(f"  {key:10s} {score_text:>9s}    ({succ}/{tot})")

    present_scores = [dim_scores[CATEGORY_TO_SCORE_KEY[c]] for c in CATEGORY_ORDER]
    if any(v is None for v in present_scores):
        overall_mean_7d = None
    else:
        overall_mean_7d = sum(present_scores) / len(present_scores)

    weighted_num = 0.0
    weighted_den = 0
    for cat in CATEGORY_ORDER:
        key = CATEGORY_TO_SCORE_KEY[cat]
        score = dim_scores[key]
        count = by_cat_total[cat]
        if score is None or count <= 0:
            continue
        weighted_num += score * count
        weighted_den += count
    overall_weighted = (weighted_num / weighted_den) if weighted_den > 0 else None

    overall_primary = overall_weighted if use_weighted_overall else overall_mean_7d
    overall_primary_label = "overall_weighted" if use_weighted_overall else "overall_mean_7d"
    overall_primary_text = "null" if overall_primary is None else f"{overall_primary:.4f}"
    print(f"  {overall_primary_label:10s} {overall_primary_text:>9s}")

    ref_label = "overall_mean_7d" if use_weighted_overall else "overall_weighted"
    ref_value = overall_mean_7d if use_weighted_overall else overall_weighted
    ref_text = "null" if ref_value is None else f"{ref_value:.4f}"
    print(f"  {ref_label:10s} {ref_text:>9s}    (reference)")

    print("")
    print("Suite Success (%):")
    for suite in sorted(by_suite_total.keys()):
        succ = by_suite_success[suite]
        tot = by_suite_total[suite]
        score = _rate(succ, tot)
        print(f"  {suite:14s} {score:.4f}    ({succ}/{tot})")

    print("")
    print(f"Episodes parsed : {len(episodes)}")
    print(f"Episodes matched: {len(matched)}")
    print(f"Episodes unknown: {len(unmatched)}")

    if unmatched:
        print("")
        print("Unknown task examples (first 10):")
        for rec in unmatched[:10]:
            print(f"  [{rec.source_log}:{rec.line_no}] suite={rec.suite} task={rec.task_text}")

    return {
        "episodes_parsed": len(episodes),
        "episodes_matched": len(matched),
        "episodes_unknown": len(unmatched),
        "dimension_scores": dim_scores,
        "overall_score": overall_primary,
        "overall_score_type": overall_primary_label,
        "overall_weighted": overall_weighted,
        "overall_mean_7d": overall_mean_7d,
        "dimension_counts": {
            CATEGORY_TO_SCORE_KEY[cat]: {"successes": by_cat_success[cat], "total": by_cat_total[cat]}
            for cat in CATEGORY_ORDER
        },
        "suite_scores": {
            suite: {
                "success_rate": _rate(by_suite_success[suite], by_suite_total[suite]),
                "successes": by_suite_success[suite],
                "total": by_suite_total[suite],
            }
            for suite in sorted(by_suite_total.keys())
        },
    }


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.task_classification:
        tc_path = pathlib.Path(args.task_classification).expanduser().resolve()
    else:
        found = _find_task_classification_default()
        if found is None:
            print(
                "Could not auto-find task_classification.json. Please pass --task-classification.",
                file=sys.stderr,
            )
            return 2
        tc_path = found

    if not tc_path.exists():
        print(f"task_classification.json not found: {tc_path}", file=sys.stderr)
        return 2

    suite_lookup, suite_category_order = _read_task_classification(tc_path)

    episodes: List[EpisodeRecord] = []
    for p in args.logs:
        log_path = pathlib.Path(p).expanduser().resolve()
        if not log_path.exists():
            print(f"Log not found: {log_path}", file=sys.stderr)
            return 2
        episodes.extend(_parse_log(log_path, args.suite_override))

    matched: List[Tuple[EpisodeRecord, str]] = []
    unmatched: List[EpisodeRecord] = []
    unresolved_indices: List[int] = []
    for idx, rec in enumerate(episodes):
        if not rec.suite or rec.suite not in suite_lookup:
            unresolved_indices.append(idx)
            continue
        norm = _normalize_text(rec.task_text)
        cat = suite_lookup[rec.suite].get(norm)
        if cat is None:
            unresolved_indices.append(idx)
        else:
            matched.append((rec, cat))

    # Fallback for unmatched records: map by suite order.
    # This is robust to language-rewritten task text as long as task order is default.
    # For each suite, infer trials per task as total_episodes // num_tasks when divisible.
    fallback_mapped = 0
    by_suite_indices: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(episodes):
        if rec.suite:
            by_suite_indices[rec.suite].append(i)

    unresolved_set = set(unresolved_indices)
    for suite, ep_indices in by_suite_indices.items():
        if suite not in suite_category_order:
            continue
        num_tasks = len(suite_category_order[suite])
        if num_tasks <= 0:
            continue
        total_eps = len(ep_indices)
        trials_per_task = total_eps // num_tasks if total_eps % num_tasks == 0 else 1
        for local_idx, global_idx in enumerate(ep_indices):
            if global_idx not in unresolved_set:
                continue
            task_idx = local_idx // max(trials_per_task, 1)
            if 0 <= task_idx < num_tasks:
                rec = episodes[global_idx]
                cat = suite_category_order[suite][task_idx]
                matched.append((rec, cat))
                unresolved_set.remove(global_idx)
                fallback_mapped += 1

    unmatched = [episodes[i] for i in sorted(unresolved_set)]

    print(f"Using task classification: {tc_path}")
    print(f"Processed logs: {len(args.logs)}")
    if fallback_mapped > 0:
        print(f"Order-based fallback matched episodes: {fallback_mapped}")
    summary = _print_summary(
        episodes=episodes,
        matched=matched,
        unmatched=unmatched,
        use_weighted_overall=(args.overall == "weighted"),
    )

    if args.json_out:
        out_path = pathlib.Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved JSON summary: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
