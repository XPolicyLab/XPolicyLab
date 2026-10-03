"""Validate and install a frozen PhysicalRSI harness bundle into XPolicyLab."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_BUNDLE = Path(__file__).parent / "releases/robodojo-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(bundle: Path) -> dict:
    bundle = bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    required = [manifest["system"]["agent_planner"].split(":", 1)[0], "memory/capability_memory.json", "harness_state/state.json"]
    required.extend(manifest["components"]["skills"])
    required.append(manifest["components"]["provenance"])
    required.append(manifest["components"]["compositions"])
    required.append(manifest["components"]["runtime_source"])
    required.append(manifest["components"]["assets"])
    missing = [item for item in required if not (bundle / item).is_file()]
    if missing:
        raise ValueError(f"Bundle is missing files: {missing}")
    if manifest.get("state") != "frozen":
        raise ValueError("Only frozen bundles may be installed")
    if manifest.get("protocol_mode") != "task_aware_skill_library":
        raise ValueError("Bundle requires the task-aware skill library protocol")
    for path in manifest["components"]["skills"]:
        descriptor = json.loads((bundle / path).read_text())
        if descriptor.get("schema") != "physicalrsi.execution-skill-descriptor/v1":
            raise ValueError("Versioned skill descriptors required")
    source = bundle / "skills/source"
    for name, expected in json.loads((source / "files.sha256.json").read_text()).items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()) or digest(path) != expected:
            raise ValueError("Skill source checksum mismatch: " + name)
    return manifest


def install(checkout: str | Path, *, bundle: str | Path = DEFAULT_BUNDLE) -> Path:
    checkout = Path(checkout).resolve()
    if not (checkout / "model_template.py").is_file():
        raise ValueError("Expected an XPolicyLab checkout")
    source = Path(bundle).resolve()
    validate(source)
    destination = checkout / "policy/physicalRSI/harness"
    if destination.exists():
        for incoming in source.rglob("*"):
            if "compatibility" in incoming.relative_to(source).parts:
                continue
            if incoming.is_file():
                target = destination / incoming.relative_to(source)
                if target.exists() and digest(target) != digest(incoming):
                    raise FileExistsError(f"Refusing to replace changed bundle file: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("compatibility", "__pycache__", "*.pyc"))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    print(install(args.checkout, bundle=args.bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
