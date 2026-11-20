#!/usr/bin/env python

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve per-task RoboTwin eval settings.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--eval-config", type=str, default="")
    parser.add_argument("--default-infer-horizon", type=str, required=True)
    parser.add_argument("--default-binarize-gripper", type=str, required=True)
    return parser.parse_args()


def load_task_names(project_root: Path) -> list[str]:
    inference_path = project_root / "evaluation" / "RoboTwin" / "inference.py"
    module = ast.parse(inference_path.read_text(encoding="utf-8"), filename=str(inference_path))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TASK_NAMES":
                task_names = ast.literal_eval(node.value)
                if not isinstance(task_names, list):
                    raise TypeError("TASK_NAMES is not a list")
                return [str(item) for item in task_names]
    raise RuntimeError(f"Failed to find TASK_NAMES in {inference_path}")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(value)
    except ValueError:
        return value.strip("'\"")


def parse_simple_task_yaml(path: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line[:1].strip():
            if not line.endswith(":"):
                raise ValueError(f"Unsupported top-level YAML line: {raw_line!r}")
            current_key = line[:-1].strip().strip("'\"")
            data[current_key] = {}
            continue
        if current_key is None or ":" not in line:
            raise ValueError(f"Unsupported nested YAML line: {raw_line!r}")
        key, value = line.split(":", 1)
        data[current_key][key.strip()] = parse_scalar(value)
    return data


def load_eval_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    try:
        from omegaconf import OmegaConf

        payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        return {} if payload is None else dict(payload)
    except ImportError:
        return parse_simple_task_yaml(config_path)


def normalize_bool(value: Any, field_name: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return "true"
    if lowered in {"0", "false", "no", "n", "off"}:
        return "false"
    raise ValueError(f"{field_name} must be boolean-like, got {value!r}")


def normalize_positive_int(value: Any, field_name: str) -> str:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got {parsed}")
    return str(parsed)


def resolve_task_settings(
    *,
    project_root: Path,
    eval_config: Path | None,
    default_infer_horizon: str,
    default_binarize_gripper: str,
) -> list[tuple[int, str, str, str]]:
    task_names = load_task_names(project_root)
    default_infer_horizon = normalize_positive_int(default_infer_horizon, "INFER_HORIZON")
    default_binarize_gripper = normalize_bool(default_binarize_gripper, "BINARIZE_GRIPPER")

    config = load_eval_config(eval_config)
    if not isinstance(config, dict):
        raise TypeError(f"RoboTwin eval config must be a mapping, got {type(config).__name__}")

    task_name_set = set(task_names)
    unknown_task_names: list[str] = []
    exact_entries: dict[str, dict[str, Any]] = {}
    for raw_key, raw_entry in config.items():
        key = str(raw_key)
        if key not in task_name_set:
            unknown_task_names.append(key)
        if not isinstance(raw_entry, dict):
            raise TypeError(f"RoboTwin eval config entry for {key!r} must be a mapping")
        exact_entries[key] = dict(raw_entry)

    if unknown_task_names:
        valid_names = ", ".join(task_names)
        unknown_names = ", ".join(sorted(unknown_task_names))
        raise ValueError(
            "Unknown RoboTwin eval config task entries: "
            f"{unknown_names}. Task names must exactly match inference.py TASK_NAMES. "
            f"Valid task names: {valid_names}"
        )

    rows: list[tuple[int, str, str, str]] = []
    for task_idx, task_name in enumerate(task_names):
        entry = exact_entries.get(task_name)

        if entry is None:
            infer_horizon = default_infer_horizon
            binarize_gripper = default_binarize_gripper
        else:
            infer_horizon = normalize_positive_int(
                entry.get("infer_horizon", default_infer_horizon),
                f"{task_name}.infer_horizon",
            )
            binarize_gripper = normalize_bool(
                entry.get("binarize_gripper", default_binarize_gripper),
                f"{task_name}.binarize_gripper",
            )

        rows.append((task_idx, task_name, infer_horizon, binarize_gripper))

    return rows


def main() -> None:
    args = parse_args()
    config_path = Path(args.eval_config) if args.eval_config.strip() else None
    rows = resolve_task_settings(
        project_root=args.project_root,
        eval_config=config_path,
        default_infer_horizon=args.default_infer_horizon,
        default_binarize_gripper=args.default_binarize_gripper,
    )
    for task_idx, task_name, infer_horizon, binarize_gripper in rows:
        print(f"{task_idx}\t{task_name}\t{infer_horizon}\t{binarize_gripper}")


if __name__ == "__main__":
    main()
