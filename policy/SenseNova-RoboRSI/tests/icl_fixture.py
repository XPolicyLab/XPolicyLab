"""Build a tiny, genuinely sealed public-training package for offline tests."""

from __future__ import annotations

import base64
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

FIXTURE_BUNDLE_ID = "offline_public_demo"
FIXTURE_EXPERIENCE_TEXT = (
    "HISTORICAL_EXPERIENCE_SENTINEL: from robot import act, reset; "
    "historical repeat=5 and boolean grippers.\n"
)
FIXTURE_IMAGE_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfV0AAAAASUVORK5CYII="
)


def _write_json(path: Path, document: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_icl_fixture(
    policy_dir: Path, task_name: str = "general_pickup"
) -> dict[str, str]:
    """Write real tiny ICL inputs and update the private submission's ICL pins.

    Production files are never selected implicitly. Callers must pass their
    temporary policy directory. No source-inventory payload, model or simulator
    is needed by the production loader.
    """
    from roborsi.components.icl import build_package

    policy_dir = Path(policy_dir)
    policy_dir.mkdir(parents=True, exist_ok=True)
    upstream = {
        "repository": "https://github.com/SenseTime-FVG/RoboRSI.git",
        "commit": "cfa591575172b638a8436fa95db28e9469785c9c",
        "inventory_path": "icl/source-inventory.json",
        "inventory_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
    }
    package_root = policy_dir / "runtime/icl/packages" / FIXTURE_BUNDLE_ID
    with tempfile.TemporaryDirectory(
        prefix=".icl-fixture-", dir=policy_dir
    ) as temporary:
        material = Path(temporary) / "material"
        (material / "skills/public_demo").mkdir(parents=True)
        (material / "evidence").mkdir()
        (material / "START_HERE.md").write_text(
            "Public training study. Read skills/public_demo/SKILL.md and "
            "evidence/reference.png for historical evidence.\n"
        )
        (material / "skills/public_demo/SKILL.md").write_text(FIXTURE_EXPERIENCE_TEXT)
        (material / "evidence/reference.png").write_bytes(FIXTURE_IMAGE_BYTES)
        (material / "evidence/geometry.json").write_text(
            '{"example_position": [0, 0, 0]}\n'
        )
        package = build_package(
            material,
            package_root,
            package_id=FIXTURE_BUNDLE_ID,
            metadata={"upstream": upstream},
        )
    manifest = json.loads((package_root / "manifest.json").read_text())
    count = len(manifest["files"])
    size = sum(
        (package_root / "payload" / name).stat().st_size for name in manifest["files"]
    )
    archive_path = policy_dir / "assets/icl" / f"{FIXTURE_BUNDLE_ID}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(package_root / "manifest.json", arcname="manifest.json")
        archive.add(package_root / "payload", arcname="payload")
    archive_size = archive_path.stat().st_size
    catalog = {
        "schema_version": 1,
        "source_commit": upstream["commit"],
        "source_inventory_sha256": upstream["inventory_sha256"],
        "source_manifest_sha256": upstream["source_manifest_sha256"],
        "official_source_commit": "726e9aabfaa642203722eb126f5eaf0f37f3e1ad",
        "tasks": {task_name: FIXTURE_BUNDLE_ID},
        "task_mapping_details": {
            task_name: {
                "kind": "original_preparation",
                "base_task": task_name,
                "source_instances": [task_name + "-public-demo-fixture"],
                "reason": "Offline fixture from public training references, no evaluation history.",
                "evidence": [],
            }
        },
        "bundles": {
            FIXTURE_BUNDLE_ID: {
                "source_files": manifest["files"],
                "file_count": count,
                "size": size,
                "entry_points": ["START_HERE.md", "skills/public_demo/SKILL.md"],
            }
        },
    }
    artifacts = {
        "schema_version": 1,
        "upstream": upstream,
        "package_format": {
            "schema_version": 1,
            "package_type": "rpent.icl.context",
            "exports": "none",
        },
        "artifacts": {
            FIXTURE_BUNDLE_ID: {
                "archive_path": f"assets/icl/{FIXTURE_BUNDLE_ID}.tar.gz",
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "archive_size": archive_size,
                "manifest_sha256": package.manifest_sha256,
                "payload_files": count,
                "payload_bytes": size,
                "url": None,
            }
        },
        "totals": {
            "bundles": 1,
            "files": count,
            "bytes": size,
            "archive_bytes": archive_size,
        },
    }
    pins = {
        "catalog_path": "icl/catalog.json",
        "catalog_sha256": _write_json(policy_dir / "icl/catalog.json", catalog),
        "artifacts_path": "icl/artifacts.json",
        "artifacts_sha256": _write_json(policy_dir / "icl/artifacts.json", artifacts),
    }
    submission_path = policy_dir / "submission.json"
    submission = (
        json.loads(submission_path.read_text()) if submission_path.is_file() else {}
    )
    submission.setdefault("schema_version", 1)
    submission["icl"] = pins
    _write_json(submission_path, submission)
    return pins
