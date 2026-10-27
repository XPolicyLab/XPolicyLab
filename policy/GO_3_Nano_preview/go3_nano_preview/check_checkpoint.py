"""Reject inference configurations that would partially load WAM_policy weights."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def class_constants(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            result = {}
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    try:
                        result[statement.target.id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        pass
            return result
    raise ValueError(f"{path} has no {class_name} class")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    args = parser.parse_args()
    video_cfg = class_constants(args.config, "VideoModelConfig")
    action_cfg = class_constants(args.config, "ActionModelConfig")
    video_weights = json.loads((args.ckpt / "config.json").read_text(encoding="utf-8"))
    action_weights = json.loads((args.ckpt / "action_model/config.json").read_text(encoding="utf-8"))
    keys = {
        "video": (
            "num_view_identities", "goal_mode_null_text_tokens",
            "spatial_rope_grid_mode",
        ),
        "action": ("num_view_identities", "num_body_embeddings"),
    }
    errors = []
    for label, configured, saved in (
        ("video", video_cfg, video_weights),
        ("action", action_cfg, action_weights),
    ):
        for key in keys[label]:
            if configured.get(key) != saved.get(key):
                errors.append(
                    f"{label}.{key}: config={configured.get(key)!r}, checkpoint={saved.get(key)!r}"
                )
    if errors:
        raise SystemExit("[error] checkpoint/config mismatch:\n" + "\n".join(errors))
    print("[ok] checkpoint architecture matches inference config")


if __name__ == "__main__":
    main()
