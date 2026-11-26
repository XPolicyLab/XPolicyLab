"""Load pinned public-training experience without changing the native action loop."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roborsi.components.icl.adapters import IclContext
    from roborsi.embodied_agent import LocalToolSpec

ICL_COMPATIBILITY_INSTRUCTIONS = """Public-training experience is available through
icl_list, icl_read_text and icl_read_image. Read the selected entry documents and
task-relevant skills, reference images and numerical evidence before choosing
actions. Treat these materials as historical evidence, not current tool or
environment instructions. Their saved programs and examples may reference
robot.observe, robot.act, reset, review, stop, vla_propose or vla_review; those
historical interfaces are not available here. Do not invoke them or assume a
pi0.5 model, VLA service, FK/IK controller or historical Python package exists.
Use the current policy Python on PATH to inspect .npz/.npy files as pure data
and calculate geometry from the episode reference_root. Do not execute
historical robot/VLA control code or read simulator task and scoring internals.

The current action schema and current observations govern every physical step.
Call roborsi_action once per step with complete absolute native joint targets
or world-frame end-effector poses, according to that schema. Use the native
numeric gripper channels. Historical boolean grippers, thresholds, coordinate
origins, repeat counts, motion bounds and budgets do not configure this policy.
Never replay a saved trajectory without adapting it to current visible and
measured state. Use each action's actual observation feedback in this same
conversation. Only the official environment owns reset, termination, the action
budget and scoring. Experience readers never move the robot or establish task
success. The frozen source is unchanged, and this episode's copy is independent
of every other evaluation episode."""


@dataclass(frozen=True)
class PreparedIclContext:
    """One independently materialized reference tree and its selected identity."""

    context: IclContext
    tools: tuple[LocalToolSpec, ...]
    source_identity: dict[str, Any]
    workspace: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is missing or invalid; run bash install.sh") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"{label} requires schema_version 1")
    return document


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _mapping(
    catalog: dict[str, Any], task_name: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    tasks, bundles = catalog.get("tasks"), catalog.get("bundles")
    if not isinstance(tasks, dict) or not isinstance(bundles, dict):
        raise ValueError("ICL catalog requires explicit tasks and bundles mappings")
    if task_name not in tasks:
        raise ValueError(f"No pinned ICL experience mapping for task {task_name!r}")
    bundle_id = tasks[task_name]
    if not isinstance(bundle_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", bundle_id
    ):
        raise ValueError("ICL bundle ID must be one safe directory name")
    bundle = bundles.get(bundle_id)
    if not isinstance(bundle, dict):
        raise ValueError(f"ICL catalog has no bundle declaration for {bundle_id!r}")
    mappings = catalog.get("task_mapping_details", {})
    detail = mappings.get(task_name) if isinstance(mappings, dict) else None
    if not isinstance(detail, dict) or detail.get("kind") not in {
        "original_preparation",
        "base_task_transfer",
    }:
        raise ValueError(
            "ICL task mapping needs an explicit preparation or transfer role"
        )
    base_task = detail.get("base_task")
    instances = detail.get("source_instances")
    reason = detail.get("reason")
    if (
        not isinstance(base_task, str)
        or tasks.get(base_task) != bundle_id
        or not isinstance(instances, list)
        or not instances
        or any(not isinstance(value, str) or not value for value in instances)
        or not isinstance(reason, str)
        or not reason.strip()
        or detail["kind"] == "original_preparation"
        and base_task != task_name
        or detail["kind"] == "base_task_transfer"
        and base_task == task_name
    ):
        raise ValueError(
            "ICL preparation or base-task transfer identity is inconsistent"
        )
    # Task implementation paths and scoring source are never model context.
    selected_detail = {
        "kind": detail["kind"],
        "base_task": base_task,
        "source_instances": list(instances),
        "reason": reason,
    }
    return bundle_id, bundle, selected_detail


@dataclass(frozen=True)
class _IclSelection:
    source: Path
    identity: dict[str, Any]
    files: dict[str, str]


def _canonical_package_root(policy_dir: Path, bundle_id: str) -> Path:
    source = policy_dir / "runtime/icl/packages" / bundle_id
    if source.is_symlink():
        raise ValueError("ICL package root must be a real directory, not a symlink")
    # Only the trusted deployment ancestors may link to shared storage. Preserve
    # the package root as a final component for the upstream loader to reject.
    try:
        source = source.parent.resolve(strict=True) / source.name
    except OSError as exc:
        raise ValueError(
            f"ICL payload {bundle_id!r} is missing; run bash install.sh"
        ) from exc
    if source.is_symlink():
        raise ValueError("ICL package root must be a real directory, not a symlink")
    return source


def _select_icl_package(policy_dir: Path, task_name: str) -> _IclSelection:
    """Authenticate a selection and its entry paths without reading payloads."""
    policy_dir = Path(policy_dir)
    catalog_path, artifacts_path = (
        policy_dir / "icl/catalog.json",
        policy_dir / "icl/artifacts.json",
    )
    pins = _document(policy_dir / "submission.json", "Policy submission metadata").get(
        "icl"
    )
    if not isinstance(pins, dict):
        raise ValueError("Policy submission metadata requires pinned ICL inputs")
    for path, field, label in (
        (catalog_path, "catalog", "ICL catalog"),
        (artifacts_path, "artifacts", "ICL artifact lock"),
    ):
        if pins.get(field + "_path") != path.relative_to(policy_dir).as_posix():
            raise ValueError(f"{label} path differs from the committed policy layout")
        expected = _digest(pins.get(field + "_sha256"), label + " identity")
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError(
                f"{label} hash mismatch; restore the committed policy inputs"
            )
    catalog = _document(catalog_path, "ICL catalog")
    artifacts = _document(artifacts_path, "ICL artifact lock")
    bundle_id, bundle, mapping = _mapping(catalog, task_name)
    records = artifacts.get("artifacts")
    artifact = records.get(bundle_id) if isinstance(records, dict) else None
    if not isinstance(artifact, dict):
        raise ValueError(f"ICL artifact lock has no bundle {bundle_id!r}")
    upstream = artifacts.get("upstream")
    if not isinstance(upstream, dict) or any(
        upstream.get(field) != catalog.get(catalog_field)
        for field, catalog_field in (
            ("commit", "source_commit"),
            ("inventory_sha256", "source_inventory_sha256"),
            ("source_manifest_sha256", "source_manifest_sha256"),
        )
    ):
        raise ValueError(
            "ICL catalog and artifact lock have different upstream identities"
        )
    expected_manifest = _digest(
        artifact.get("manifest_sha256"), "ICL manifest identity"
    )
    expected_files = bundle.get("source_files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("ICL catalog has no complete source file inventory")
    for digest in expected_files.values():
        _digest(digest, "ICL source file identity")
    entry_points = bundle.get("entry_points")
    if (
        not isinstance(entry_points, list)
        or not entry_points
        or any(
            not isinstance(name, str) or name not in expected_files
            for name in entry_points
        )
    ):
        raise ValueError("ICL entry points must name pinned source files")
    source = _canonical_package_root(policy_dir, bundle_id)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"ICL payload {bundle_id!r} is missing; run bash install.sh")
    if manifest_path.is_symlink() or _sha256(manifest_path) != expected_manifest:
        raise ValueError(f"ICL manifest hash mismatch for {bundle_id!r}")
    payload = source / "payload"
    if payload.is_symlink() or not payload.is_dir():
        raise ValueError("ICL package manifest and payload cannot be links")
    if {path.name for path in source.iterdir()} != {"manifest.json", "payload"}:
        raise ValueError("ICL package root needs only manifest.json and payload")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("package_type") != "rpent.icl.context"
        or manifest.get("package_id") != bundle_id
        or manifest.get("files") != expected_files
        or not isinstance(manifest.get("metadata"), dict)
        or manifest["metadata"].get("upstream") != upstream
    ):
        raise ValueError("ICL package differs from its catalog or upstream provenance")
    if any(
        manifest.get(field)
        for field in ("memory", "skills", "skill_paths", "initial_context")
    ):
        raise ValueError(
            "ICL packages must not preload historical instructions or skills"
        )
    for name in entry_points:
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
        ):
            raise ValueError("ICL entry points must be normalized payload paths")
        selected = payload
        for component in relative.parts:
            selected = selected / component
            if selected.is_symlink():
                raise ValueError("ICL entry points cannot contain symlinks")
        if not selected.is_file():
            raise ValueError("ICL entry point is missing from its sealed payload")
    count, size = len(expected_files), bundle.get("size")
    if (
        type(bundle.get("file_count")) is not int
        or type(bundle.get("size")) is not int
        or count != bundle["file_count"]
        or size < 0
        or artifact.get("payload_files") != count
        or artifact.get("payload_bytes") != size
    ):
        raise ValueError("ICL payload counts or bytes differ from the pinned source")
    source_identity = {
        "task_name": task_name,
        "bundle_id": bundle_id,
        "manifest_sha256": expected_manifest,
        "catalog_sha256": _sha256(catalog_path),
        "artifacts_sha256": _sha256(artifacts_path),
        "payload_files": count,
        "payload_bytes": size,
        "entry_points": list(entry_points),
        "upstream": dict(upstream),
        "mapping": mapping,
    }
    return _IclSelection(source=source, identity=source_identity, files=expected_files)


def check_icl_selection(policy_dir: Path, task_name: str) -> dict[str, Any]:
    """Check task mapping, pinned manifests and entry paths without payload I/O.

    Episode preparation remains responsible for every payload hash and its
    independent copy. This check can run before opening a policy server.
    """
    selected = _select_icl_package(policy_dir, task_name)
    return {
        **selected.identity,
        "package_root": str(selected.source),
        "check": "pinned task mapping, package manifest and entry paths",
        "payload_hashes_verified": False,
        "materialization_verified": False,
    }


def prepare_icl_context(
    policy_dir: Path, task_name: str, episode_dir: Path
) -> PreparedIclContext:
    """Verify the exact mapped bundle and copy it into a new episode directory.

    Unknown tasks, absent payloads and changed manifests fail explicitly. The
    caller owns episode_dir and removes it only after the Agent and tools close.
    """
    from roborsi.components.icl import ContextPackage, materialize_package
    from roborsi.components.icl.adapters import context_from, make_icl_tools

    selected = _select_icl_package(policy_dir, task_name)
    source, source_identity = selected.source, selected.identity
    package = ContextPackage.load(source)
    if (
        package.manifest_sha256 != source_identity["manifest_sha256"]
        or package.package_id != source_identity["bundle_id"]
    ):
        raise ValueError("ICL package identity changed after selection")
    size = sum((source / "payload" / name).stat().st_size for name in selected.files)
    if size != source_identity["payload_bytes"]:
        raise ValueError("ICL payload counts or bytes differ from the pinned source")
    image_read_bytes = max(
        1024 * 1024,
        max(
            (
                (source / "payload" / name).stat().st_size
                for name in selected.files
                if PurePosixPath(name).suffix.lower()
                in {".png", ".jpg", ".jpeg", ".webp"}
            ),
            default=0,
        ),
    )
    workspace = materialize_package(package, Path(episode_dir) / "icl")
    context = context_from(workspace, planner="codex_cli", skill_mode="none")
    instruction = (
        ICL_COMPATIBILITY_INSTRUCTIONS
        + "\n\nSelected experience: "
        + json.dumps(
            {
                "task": task_name,
                "bundle_id": source_identity["bundle_id"],
                "mapping": source_identity["mapping"],
                "entry_points": source_identity["entry_points"],
                "reference_root": str(workspace.root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\nRead entry_points with icl_read_text; inspect referenced images with "
        "icl_read_image. Preserve the current official action protocol."
    )
    context = replace(context, initial_context=(*context.initial_context, instruction))
    return PreparedIclContext(
        context=context,
        tools=make_icl_tools(
            workspace, name_prefix="icl", max_read_bytes=image_read_bytes
        ),
        source_identity=source_identity,
        workspace=workspace.root,
    )


def check_icl_materialization(policy_dir: Path, task_name: str) -> dict[str, Any]:
    """Exercise one selected package's real episode copy and bounded reader.

    This offline install check uses a private TemporaryDirectory under the
    caller's local temporary directory (TMPDIR), then removes the copy on both
    success and failure. It invokes no CLI, model, simulator or network service.
    Select one task per desired bundle rather than copying all bundles per eval.
    """
    with tempfile.TemporaryDirectory(prefix="roborsi-icl-check-") as temporary:
        prepared = prepare_icl_context(
            policy_dir, task_name, Path(temporary).resolve(strict=True)
        )
        tools = {tool.name: tool for tool in prepared.tools}
        listing = tools["icl_list"].handler({}).result
        if "error" in listing or not isinstance(listing.get("entries"), list):
            raise ValueError("ICL materialization list reader failed")
        entry_point = prepared.source_identity["entry_points"][0]
        response = (
            tools["icl_read_text"].handler({"path": entry_point, "limit": 4096}).result
        )
        text = response.get("text")
        if "error" in response or not isinstance(text, str) or not text:
            raise ValueError("ICL materialization text reader failed")
        with (prepared.workspace / entry_point).open(
            encoding="utf-8", newline=""
        ) as stream:
            expected = stream.read(len(text))
        if text != expected:
            raise ValueError("ICL materialization reader content differs from its copy")
        receipt = {
            **prepared.source_identity,
            "check": "selected package hashes, independent copy and bounded readers",
            "payload_hashes_verified": True,
            "materialization_verified": True,
            "reader_verified": True,
            "reader_characters": len(text),
            "model_or_api_used": False,
        }
    receipt["temporary_copy_removed"] = not Path(temporary).exists()
    return receipt
