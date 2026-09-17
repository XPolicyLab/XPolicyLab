"""Explicitly prepare a RoboDojo checkout for the verified Full35 simulator setup.

This helper is never invoked by install.sh. It recognizes only the tested
material source files, preserves HF snapshot MDL names, and optionally selects
one simulation environment. It does not validate model performance or assets.
"""

from __future__ import annotations

import argparse
import ast
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

import yaml


PATCHES = {
    "env/scene_manager/objects/ground.py": (
        "12dd51addae6f5b7c6ad410b4ae7248231d216cf63513c564057d577a09a2972",
        (
            (b"return [str(path.resolve())]", b"return [str(path.absolute())]"),
            (b'return [str(p.resolve()) for p in path.glob("**/*.mdl")]', b'return [str(p.absolute()) for p in path.glob("**/*.mdl")]'),
        ),
    ),
    "env/scene_manager/objects/table.py": (
        "c0e71396b0d53a250f74c75490e91086d67962237400a49f886a13ec07875308",
        ((b"return str(p.resolve())", b"return str(p.absolute())"),),
    ),
}
MAPPING_PATH = "env_cfg/arx_x5.yml"
STATE_NAME = ".kinrt_full35_preparation.json"
STATE_VERSION = 1


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def scoped_file(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Expected a relative file inside the RoboDojo root: {relative}")
    path = root / relative_path
    if not path.resolve(strict=must_exist).is_relative_to(root):
        raise ValueError(f"File escapes the RoboDojo root: {path}")
    for component in (path, *path.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError(f"Preparation targets must not use symbolic links: {component}")
    if (must_exist or path.exists()) and not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return path


def material_plan(root: Path, mode: str) -> list[dict]:
    plan = []
    for relative, (original_sha, replacements) in PATCHES.items():
        path = scoped_file(root, relative)
        current = path.read_bytes()
        if sha256(current) == original_sha:
            original = current
            patched = current
            for old, new in replacements:
                if patched.count(old) != 1:
                    raise ValueError(f"Patch anchor mismatch: {relative}")
                patched = patched.replace(old, new, 1)
        else:
            original = current
            for old, new in reversed(replacements):
                if original.count(new) != 1:
                    raise ValueError(f"Unknown material source: {relative} (SHA-256 {sha256(current)})")
                original = original.replace(new, old, 1)
            if sha256(original) != original_sha:
                raise ValueError(f"Unknown material source: {relative} (SHA-256 {sha256(current)})")
            patched = current
        ast.parse(patched.decode("utf-8"), filename=str(path))
        plan.append({
            "path": path, "relative": relative, "current": current,
            "desired": original if mode == "revert" else patched,
            "original_sha256": original_sha, "patched_sha256": sha256(patched),
            "prior_state": "original" if current == original else "patched",
        })
    return plan


def simulation_config(content: bytes, path: Path) -> dict:
    config = yaml.safe_load(content)
    if not isinstance(config, dict) or not isinstance(config.get("scene"), dict):
        raise ValueError(f"Expected a scene mapping in {path}")
    count = config["scene"].get("num_envs")
    if type(count) is not int or count <= 0:
        raise ValueError(f"scene.num_envs must be a positive integer in {path}")
    return config


def set_num_envs(config: dict, count: int) -> bytes:
    updated = deepcopy(config)
    updated["scene"] = dict(updated["scene"])
    updated["scene"]["num_envs"] = count
    return yaml.safe_dump(updated, sort_keys=False).encode("utf-8")


def config_plan(root: Path, mode: str, num_envs: int | None) -> tuple[dict, dict | None, Path]:
    mapping_path = scoped_file(root, MAPPING_PATH)
    mapping_content = mapping_path.read_bytes()
    mapping = yaml.safe_load(mapping_content)
    config_mapping = mapping.get("config") if isinstance(mapping, dict) else None
    sim_name = config_mapping.get("sim") if isinstance(config_mapping, dict) else None
    if not isinstance(sim_name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", sim_name) is None:
        raise ValueError(f"Expected a simple config.sim name in {mapping_path}")
    relative = f"env_cfg/sim/{sim_name}.yml"
    path = scoped_file(root, relative)
    current = path.read_bytes()
    config = simulation_config(current, path)
    desired = current
    state_path = scoped_file(root, STATE_NAME, must_exist=False)
    state = None
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            raise ValueError(f"Unknown preparation state: {state_path}")
        if state.get("sim_path") != relative or state.get("mapping_sha256") != sha256(mapping_content):
            raise ValueError("The ARX-X5 simulator mapping changed after preparation; no files were changed.")
        try:
            original = base64.b64decode(state["original_yaml_base64"], validate=True)
            original_config = simulation_config(original, path)
            applied = set_num_envs(original_config, 1)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid configuration backup in {state_path}") from error
        if sha256(original) != state.get("original_sha256") or sha256(applied) != state.get("applied_sha256"):
            raise ValueError(f"Configuration backup hashes do not match in {state_path}")
        if sha256(current) not in (state["original_sha256"], state["applied_sha256"]):
            raise ValueError("The simulation YAML changed after preparation; no files were changed.")
        if mode == "revert":
            desired = original
        elif num_envs is not None:
            desired = applied
    elif num_envs is not None and config["scene"]["num_envs"] != num_envs:
        desired = set_num_envs(config, num_envs)
        state = {
            "version": STATE_VERSION, "sim_path": relative,
            "mapping_sha256": sha256(mapping_content),
            "original_sha256": sha256(current), "applied_sha256": sha256(desired),
            "original_yaml_base64": base64.b64encode(current).decode("ascii"),
        }
    return {
        "path": path, "relative": relative, "current": current, "desired": desired,
        "mapping_content": mapping_content,
        "mapping_sha256": sha256(mapping_content), "num_envs_before": config["scene"]["num_envs"],
        "num_envs_target": simulation_config(desired, path)["scene"]["num_envs"],
    }, state, state_path


def atomic_write(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".kinrt-", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def prepare_robodojo(root: Path, mode: str = "check", num_envs: int | None = None) -> dict:
    if mode not in ("check", "apply", "revert"):
        raise ValueError(f"Unknown preparation mode: {mode}")
    if num_envs is not None and (type(num_envs) is not int or num_envs != 1):
        raise ValueError("Only the verified --num-envs 1 setup is supported.")
    if mode == "revert" and num_envs is not None:
        raise ValueError("Revert restores the recorded configuration; omit --num-envs.")
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Expected a RoboDojo directory: {root}")
    materials = material_plan(root, mode)
    simulation, state, state_path = config_plan(root, mode, num_envs)
    plan = [*materials, simulation]
    if mode != "check":
        # Validate every source and configuration before writing any target.
        if scoped_file(root, MAPPING_PATH).read_bytes() != simulation["mapping_content"]:
            raise ValueError("The ARX-X5 simulator mapping changed during preparation.")
        for item in plan:
            if item["path"].read_bytes() != item["current"]:
                raise ValueError(f"File changed during preparation: {item['path']}")
        if mode == "apply" and state is not None and not state_path.exists():
            atomic_write(state_path, (json.dumps(state, indent=2) + "\n").encode("utf-8"))
        for item in plan:
            if item["current"] != item["desired"]:
                atomic_write(item["path"], item["desired"])
        if mode == "revert" and state_path.exists():
            state_path.unlink()
    return {
        "mode": mode, "robodojo_root": str(root),
        "material_sources": [
            {
                "path": item["relative"], "prior_state": item["prior_state"],
                "original_sha256": item["original_sha256"], "patched_sha256": item["patched_sha256"],
                "before_sha256": sha256(item["current"]),
                "after_sha256": sha256(item["current"] if mode == "check" else item["desired"]),
                "would_change": item["current"] != item["desired"],
                "written": mode != "check" and item["current"] != item["desired"],
            }
            for item in materials
        ],
        "simulation": {
            "mapping_path": MAPPING_PATH, "mapping_sha256": simulation["mapping_sha256"],
            "path": simulation["relative"], "num_envs_before": simulation["num_envs_before"],
            "num_envs_target": simulation["num_envs_target"],
            "before_sha256": sha256(simulation["current"]),
            "after_sha256": sha256(simulation["current"] if mode == "check" else simulation["desired"]),
            "would_change": simulation["current"] != simulation["desired"],
            "written": mode != "check" and simulation["current"] != simulation["desired"],
            "backup_file": STATE_NAME if state_path.exists() else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", type=Path, required=True, help="Explicit simulator checkout; stop it before applying or reverting.")
    parser.add_argument("--mode", choices=("check", "apply", "revert"), default="check")
    parser.add_argument("--num-envs", type=int, choices=(1,), help="Optionally select the verified single-environment setup.")
    args = parser.parse_args()
    try:
        report = prepare_robodojo(args.robodojo_root, args.mode, args.num_envs)
    except (OSError, ValueError, SyntaxError, yaml.YAMLError) as error:
        parser.exit(1, f"[KinRT][ERROR] {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
