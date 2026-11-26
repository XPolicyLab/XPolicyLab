"""Paper tables from the results store: main, ablation, cost and learning curve (Markdown + CSV).

Every table has a ``<name>.cells.json`` sidecar mapping each cell (``row key|column``) to the run ids it
was computed from, so every number in the paper traces back to runs.

    python -m knowin_agent.experiments.make_tables --results results.jsonl --out tables/ [--baseline full]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from knowin_agent.experiments import results as store
from knowin_agent.experiments import switches as registry
from knowin_agent.experiments.splits import TEST_DEV
from knowin_agent.knowledge import DIMENSIONS, task_dimension

Records = list[dict]


@dataclass
class Table:
    name: str
    title: str
    columns: list[str]
    key_columns: int
    rows: list[list[Any]] = field(default_factory=list)
    cells: dict[str, list[str]] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)

    def add(self, row: Sequence[Any], sources: Mapping[str, Iterable[str]]) -> None:
        """One row; ``sources`` maps a column name (or '*' for every value column) to its run ids."""
        self.rows.append(list(row))
        key = "|".join(str(v) for v in row[:self.key_columns])
        for column in self.columns[self.key_columns:]:
            ids = sources.get(column, sources.get("*", ()))
            self.cells[f"{key}|{column}"] = sorted(set(ids))

    def csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.columns)
        for row in self.rows:
            writer.writerow(["" if v is None else (round(v, 4) if isinstance(v, float) else v) for v in row])
        return buffer.getvalue()

    def markdown(self) -> str:
        lines = [f"### {self.title}", "", "| " + " | ".join(self.columns) + " |",
                 "|" + "|".join(" --- " for _ in self.columns) + "|"]
        for row in self.rows:
            lines.append("| " + " | ".join(_fmt(column, v) for column, v in zip(self.columns, row)) + " |")
        if not self.rows:
            lines.append("| " + " | ".join("" for _ in self.columns) + " |")
        return "\n".join(lines) + "\n"

    def sidecar(self) -> dict:
        return {"table": self.name, "title": self.title, "filters": self.filters, "cells": self.cells}


def _fmt(column: str, value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if "rate" in column or column.startswith("sr") or column.endswith("_sr") or column == "delta_sr":
            return f"{100 * value:+.1f}%" if column.startswith("delta") else f"{100 * value:.1f}%"
        return f"{value:+.1f}" if column.startswith("delta") else f"{value:.1f}"
    return str(value)


def _ids(records: Iterable[Mapping]) -> list[str]:
    return [store.run_id(r) for r in records]


def _scored(records: Iterable[Mapping]) -> tuple[Records, Records]:
    """(usable, dropped): excluded runs (unstable episodes) and runs without a score are dropped."""
    usable, dropped = [], []
    for r in records:
        (dropped if (r.get("results") or {}).get("excluded") or store.score_of(r) is None else usable).append(r)
    return usable, dropped


def _success(record: Mapping) -> float:
    return 1.0 if (record.get("results") or {}).get("success") else 0.0


def _by(records: Iterable[Mapping], *keys) -> dict[tuple, Records]:
    groups: dict[tuple, Records] = defaultdict(list)
    for r in records:
        groups[tuple(k(r) for k in keys)].append(dict(r))
    return dict(groups)


def _task(r: Mapping) -> str:
    return r["identity"]["task"]


def _profile(r: Mapping) -> str:
    return str(r.get("profile"))


def _task_stats(records: Records) -> dict[str, dict]:
    """Per task: mean score, success rate, n and run ids."""
    out = {}
    for (task,), group in sorted(_by(records, _task).items()):
        out[task] = {"score": mean(store.score_of(r) for r in group), "sr": mean(_success(r) for r in group),
                     "n": len(group), "ids": _ids(group)}
    return out


def _aggregate(stats: Mapping[str, dict], tasks: Iterable[str]) -> Optional[dict]:
    """Tasks weigh equally (RoboDojo averages per-task scores)."""
    chosen = [stats[t] for t in tasks if t in stats]
    if not chosen:
        return None
    return {"score": mean(s["score"] for s in chosen), "sr": mean(s["sr"] for s in chosen),
            "n": sum(s["n"] for s in chosen), "tasks": len(chosen), "ids": [i for s in chosen for i in s["ids"]]}


def _evaluated(records: Iterable[Mapping], phase: str, split: Optional[str]) -> Records:
    return store.query(records, phase=phase, split=split)


def main_table(records: Iterable[Mapping], *, phase: str = "test", split: Optional[str] = TEST_DEV) -> Table:
    """Per profile: one row per task, per RoboDojo dimension and overall (mean score, SR, n)."""
    table = Table("main", f"Main results ({phase}, split {split or 'any'})",
                  ["profile", "level", "name", "dimension", "tasks", "mean_score", "success_rate", "n"], 3,
                  filters={"phase": phase, "split": split, "excluded": "dropped"})
    for (profile,), group in sorted(_by(_evaluated(records, phase, split), _profile).items()):
        usable, dropped = _scored(group)
        stats = _task_stats(usable)
        for task, s in stats.items():
            table.add([profile, "task", task, task_dimension(task) or "unknown", 1, s["score"], s["sr"], s["n"]],
                      {"*": s["ids"]})
        for dimension in DIMENSIONS + ("unknown",):
            members = [t for t in stats if (task_dimension(t) or "unknown") == dimension]
            agg = _aggregate(stats, members)
            if agg:
                table.add([profile, "dimension", dimension, dimension, agg["tasks"], agg["score"], agg["sr"],
                           agg["n"]], {"*": agg["ids"]})
        agg = _aggregate(stats, stats)
        if agg:
            table.add([profile, "overall", "all", "", agg["tasks"], agg["score"], agg["sr"], agg["n"]],
                      {"*": agg["ids"]})
        if dropped:
            table.cells[f"{profile}|dropped"] = sorted(_ids(dropped))
    return table


def _switches_of(records: Records) -> Optional[dict]:
    for r in records:
        if r.get("switches"):
            return r["switches"]
    return None


def _diff_text(base: Optional[dict], other: Optional[dict]) -> str:
    if base is None or other is None:
        return "unknown"
    changes = registry.diff(base, other)
    return "; ".join(f"{k}: {a}→{b}" for k, (a, b) in changes.items()) or "none"


def ablation_table(records: Iterable[Mapping], *, baseline: str = "full", phase: str = "test",
                   split: Optional[str] = TEST_DEV, per_task: bool = True) -> Table:
    """Each profile against ``baseline`` on the tasks both have runs for (overall, then per task)."""
    table = Table("ablation", f"Ablations vs {baseline} ({phase}, split {split or 'any'}; common tasks only)",
                  ["profile", "scope", "switch_diff", "tasks", "baseline_score", "score", "delta_score",
                   "baseline_sr", "sr", "delta_sr", "n_baseline", "n"], 2,
                  filters={"phase": phase, "split": split, "baseline": baseline, "excluded": "dropped"})
    groups = _by(_evaluated(records, phase, split), _profile)
    if (baseline,) not in groups:
        return table
    base_usable, _ = _scored(groups[(baseline,)])
    base_stats = _task_stats(base_usable)
    for (profile,), group in sorted(groups.items()):
        if profile == baseline:
            continue
        usable, _ = _scored(group)
        stats = _task_stats(usable)
        common = sorted(set(stats) & set(base_stats))
        if not common:
            continue
        diff = _diff_text(_switches_of(groups[(baseline,)]), _switches_of(group))
        scopes = [("all", common)] + ([(t, [t]) for t in common] if per_task else [])
        for scope, tasks in scopes:
            b, p = _aggregate(base_stats, tasks), _aggregate(stats, tasks)
            both = b["ids"] + p["ids"]
            table.add([profile, scope, diff if scope == "all" else "", len(tasks), b["score"], p["score"],
                       p["score"] - b["score"], b["sr"], p["sr"], p["sr"] - b["sr"], b["n"], p["n"]],
                      {"baseline_score": b["ids"], "baseline_sr": b["ids"], "n_baseline": b["ids"],
                       "score": p["ids"], "sr": p["ids"], "n": p["ids"], "*": both})
    return table


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _mean_of(records: Records, get) -> tuple[Optional[float], list[str]]:
    values = [(store.run_id(r), _number(get(r))) for r in records]
    known = [(i, v) for i, v in values if v is not None]
    return (mean(v for _, v in known) if known else None), [i for i, _ in known]


def _cost(r: Mapping, key: str) -> Any:
    return (r.get("costs") or {}).get(key)


def _tokens(r: Mapping, key: str) -> Any:
    tokens = _cost(r, "tokens")
    return tokens.get(key) if isinstance(tokens, dict) else None


def _process(r: Mapping, key: str) -> Any:
    value = (r.get("process") or {}).get(key)
    return value.get("total") if isinstance(value, dict) else value


def _step_usage(r: Mapping) -> Optional[float]:
    steps, limit = _number(_cost(r, "native_steps")), _number(_cost(r, "native_step_limit"))
    return steps / limit if steps is not None and limit else None


def _decisions(r: Mapping) -> Any:
    decisions = _cost(r, "decisions")
    return decisions.get("total") if isinstance(decisions, dict) else None


COST_COLUMNS = {
    "native_steps": lambda r: _cost(r, "native_steps"),
    "step_usage_rate": _step_usage,
    "executions": lambda r: _cost(r, "executions"),
    "decisions": _decisions,
    "input_tokens": lambda r: _tokens(r, "input"),
    "output_tokens": lambda r: _tokens(r, "output"),
    "wall_seconds": lambda r: _cost(r, "wall_seconds"),
    "tool_calls": lambda r: _process(r, "tool_calls"),
    "delegations": lambda r: _process(r, "delegations"),
    "async_tracks": lambda r: _process(r, "async_tracks"),
    "stalls": lambda r: _process(r, "stalls"),
    "over_budget_refusals": lambda r: _process(r, "over_budget_refusals"),
}


def cost_table(records: Iterable[Mapping]) -> Table:
    """Per profile and phase: mean of each cost / process statistic over the runs that report it."""
    table = Table("cost", "Cost and process statistics (means over runs reporting each value)",
                  ["profile", "phase", "n", *COST_COLUMNS], 2)
    for (profile, phase), group in sorted(_by(records, _profile, lambda r: r["identity"]["phase"]).items()):
        row, sources = [profile, phase, len(group)], {"n": _ids(group)}
        for column, get in COST_COLUMNS.items():
            value, ids = _mean_of(group, get)
            row.append(value)
            sources[column] = ids
        table.add(row, sources)
    return table


def learning_curve(records: Iterable[Mapping], *, split: Optional[str] = TEST_DEV) -> Table:
    """Per pipeline and task: cumulative practice executions up to round r vs the round-r test-dev score."""
    table = Table("learning_curve", f"Learning curve (split {split or 'any'})",
                  ["profile", "task", "pipeline_id", "round", "cumulative_practice_executions",
                   "test_dev_mean_score", "test_dev_success_rate", "n_test"], 4,
                  filters={"split": split, "practice_phase": "practice", "test_phase": "test"})
    records = [r for r in records if r["identity"].get("pipeline_id") and r["identity"].get("round") is not None]
    practice = _by(store.query(records, phase="practice"), _profile, _task, lambda r: r["identity"]["pipeline_id"])
    tests = _by(_scored(store.query(records, phase="test", split=split))[0], _profile, _task,
                lambda r: r["identity"]["pipeline_id"])
    for key, group in sorted(tests.items()):
        runs = practice.get(key, [])
        for (round_,), at_round in sorted(_by(group, lambda r: r["identity"]["round"]).items()):
            done = [r for r in runs if r["identity"]["round"] <= round_]
            executions = sum(_number(_cost(r, "executions")) or 0 for r in done)
            test_ids = _ids(at_round)
            table.add([*key, round_, int(executions), mean(store.score_of(r) for r in at_round),
                       mean(_success(r) for r in at_round), len(at_round)],
                      {"cumulative_practice_executions": _ids(done), "*": test_ids})
    return table


def build_tables(records: Iterable[Mapping], *, baseline: str = "full", phase: str = "test",
                 split: Optional[str] = TEST_DEV) -> list[Table]:
    records = list(records)
    return [main_table(records, phase=phase, split=split),
            ablation_table(records, baseline=baseline, phase=phase, split=split),
            cost_table(records), learning_curve(records, split=split)]


def write_tables(source: Union[str, Path, Iterable[Mapping]], out_dir: Union[str, Path], **options) -> dict[str, Path]:
    """Write ``<table>.md``, ``<table>.csv`` and ``<table>.cells.json`` for each table, plus ``tables.md``."""
    records = store.load(source) if isinstance(source, (str, Path)) else list(source)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    tables = build_tables(records, **options)
    for table in tables:
        for suffix, text in (("md", table.markdown()), ("csv", table.csv()),
                             ("cells.json", json.dumps(table.sidecar(), indent=1, ensure_ascii=False) + "\n")):
            path = out / f"{table.name}.{suffix}"
            path.write_text(text, encoding="utf-8")
            written[f"{table.name}.{suffix}"] = path
    index = out / "tables.md"
    index.write_text("\n".join(t.markdown() for t in tables), encoding="utf-8")
    written["tables.md"] = index
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m knowin_agent.experiments.make_tables", description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="results store (results.jsonl)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--baseline", default="full", help="profile the ablation table compares against")
    parser.add_argument("--phase", default="test")
    parser.add_argument("--split", default=TEST_DEV, help="scene split for main/ablation/curve ('any' = all)")
    args = parser.parse_args(argv)
    split = None if args.split == "any" else args.split
    written = write_tables(args.results, args.out, baseline=args.baseline, phase=args.phase, split=split)
    print("\n".join(str(p) for p in written.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
