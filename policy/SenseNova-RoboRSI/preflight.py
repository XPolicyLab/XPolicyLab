"""Check the pinned CLI and provider configuration without calling a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

POLICY_DIR = Path(__file__).resolve().parent
SOURCE_PATHS = {
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "roborsi",
    "robots/__init__.py",
    "robots/robodojo/agents/icl_pi05",
    "robots/robodojo/benchmark_service.py",
    "robots/robodojo/roborsi_adapter.py",
    "robots/robodojo/worker.py",
    "robots/robodojo/episode.py",
    "robots/robodojo/native.py",
    "robots/robodojo/native_files.json",
    "robots/robodojo/eef_contracts.py",
    "robots/robodojo/eef_queries.py",
    "robots/robodojo/eef_native.py",
    "robots/robodojo/eef_types.py",
    "robots/robodojo/eef_trace.py",
    "robots/robodojo/eef_docs.py",
    "robots/robodojo/eef_pose.py",
    "robots/robodojo/joint_docs.py",
    "benchmarks/package.py",
    "benchmarks/build_inputs.py",
    "benchmarks/robodojo/prepare.py",
}


def relative_artifact_path(value: Any, field: str) -> Path:
    """Keep installation artifacts within the policy directory."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"submission.json: {field} is required")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(
            f"submission.json: {field} must stay within the policy directory"
        )
    return relative


def sha256_file(path: Path) -> str:
    """Return a file digest without loading the artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_submission(policy_dir: Path = POLICY_DIR) -> dict[str, Any]:
    """Validate nonsecret submission metadata using only the standard library."""
    path = policy_dir / "submission.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("submission.json: unsupported schema_version")
    source = metadata.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("submission.json: source must be a mapping")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", ""))):
        raise ValueError("submission.json: source.commit must be a full Git commit")
    if source.get("url") != "https://github.com/SenseTime-FVG/RoboRSI.git":
        raise ValueError("submission.json: unexpected RoboRSI source repository")
    paths = source.get("paths")
    if (
        not isinstance(paths, list)
        or not all(isinstance(path, str) for path in paths)
        or len(paths) != len(SOURCE_PATHS)
        or set(paths) != SOURCE_PATHS
    ):
        raise ValueError(
            "submission.json: source.paths must contain only the required RoboDojo export"
        )
    validate_source_bundle(metadata, policy_dir)
    validate_adaptive_profile(metadata, policy_dir)
    if not metadata.get("agent_label"):
        raise ValueError("submission.json: agent_label is required")
    cli = metadata.get("cli", {})
    if not isinstance(cli, dict):
        raise ValueError("submission.json: cli must be a mapping")
    if cli.get("distribution") != "standalone":
        raise ValueError(
            "submission.json: the confirmed CLI distribution is standalone"
        )
    version = cli.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(
            "submission.json: an exact CLI version is required; never use latest"
        )
    if cli.get("expected_version") != f"codex-cli {version}":
        raise ValueError(
            "submission.json: expected_version must match the CLI --version output"
        )
    if cli.get("target") != "x86_64-unknown-linux-musl":
        raise ValueError("submission.json: unsupported confirmed standalone target")
    if not re.fullmatch(r"[0-9a-f]{64}", str(cli.get("sha256", ""))):
        raise ValueError(
            "submission.json: the confirmed CLI executable SHA-256 is required"
        )
    companions = cli.get("companion_sha256")
    if not isinstance(companions, dict) or set(companions) != {
        "bin/codex-code-mode-host",
        "codex-resources/bwrap",
    }:
        raise ValueError(
            "submission.json: confirmed Codex companion identities are required"
        )
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(digest)) for digest in companions.values()
    ):
        raise ValueError(
            "submission.json: companion identities must be SHA-256 digests"
        )
    return metadata


def validate_adaptive_profile(metadata: dict[str, Any], policy_dir: Path) -> None:
    """Verify the separately pinned adaptive snapshot, including public interfaces."""
    hybrid = metadata.get("hybrid", {})
    if hybrid.get("variant") != "adaptive_2_50":
        raise ValueError("The submission requires its explicit adaptive hybrid profile")
    root = policy_dir / relative_artifact_path(
        hybrid.get("profile_path"), "hybrid.profile_path"
    )
    manifest_path = policy_dir / relative_artifact_path(
        hybrid.get("manifest_path"), "hybrid.manifest_path"
    )
    expected = hybrid.get("manifest_sha256")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(expected))
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != expected
    ):
        raise ValueError("The adaptive source provenance is missing or changed")
    document = json.loads(manifest_path.read_text())
    if (
        document.get("baseline_commit") != metadata["source"]["commit"]
        or document.get("policy_variant") != hybrid["variant"]
    ):
        raise ValueError("Adaptive snapshot has a different baseline or policy")
    names = set()
    for row in document["files"]:
        relative = relative_artifact_path(row["path"], "adaptive source file")
        name = relative.as_posix()
        path = root / relative
        if (
            name in names
            or not path.resolve().is_relative_to(root.resolve())
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError("Adaptive profile is missing or changed: " + name)
        names.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Adaptive source cannot contain symlinks")
        if path.is_file() and not (
            "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}
        ):
            actual.add(path.relative_to(root).as_posix())
    if actual != names | {manifest_path.relative_to(root).as_posix()}:
        raise ValueError("Adaptive snapshot contains unexpected files")


def validate_archive(metadata: dict[str, Any], policy_dir: Path = POLICY_DIR) -> Path:
    """Require an authenticated installation artifact or a pinned download URL."""
    cli = metadata["cli"]
    relative = relative_artifact_path(cli.get("archive_path"), "cli.archive_path")
    archive = policy_dir / relative
    digest = cli.get("archive_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(
            "The fixed Codex native release is declared, but its install archive "
            "SHA-256 is not recorded. Supply the verified release archive and "
            "cli.archive_sha256; a Python SDK or npm version is not a substitute."
        )
    if archive.is_file():
        if sha256_file(archive) != digest:
            raise ValueError("Codex standalone archive SHA-256 mismatch")
        return archive
    download = cli.get("download", {})
    if not isinstance(download, dict):
        raise ValueError("submission.json: cli.download must be a mapping")
    url = download.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError(
            f"Codex installation archive is missing: {archive}. Record its confirmed "
            "HTTPS distribution URL and digest in cli.download or supply that archive."
        )
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "submission.json: cli.download.url must be HTTPS without credentials"
        )
    digest = download.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(
            "submission.json: cli.download.sha256 is required for the official archive"
        )
    relative = relative_artifact_path(download.get("path"), "cli.download.path")
    archive = policy_dir / relative
    if archive.is_file() and sha256_file(archive) != digest:
        raise ValueError("Cached official Codex archive SHA-256 mismatch")
    return archive


def validate_companions(metadata: dict[str, Any], codex_root: Path) -> None:
    """Check the host and sandbox shipped with the confirmed standalone CLI."""
    manifest_path = codex_root / "codex-package.json"
    if not manifest_path.is_file():
        raise ValueError("Codex standalone package manifest is missing")
    package = json.loads(manifest_path.read_text())
    if any(
        package.get(key) != value
        for key, value in {
            "version": metadata["cli"]["version"],
            "target": metadata["cli"]["target"],
            "entrypoint": "bin/codex",
        }.items()
    ):
        raise ValueError("Codex standalone package version/target/entrypoint mismatch")
    for relative, expected in metadata["cli"]["companion_sha256"].items():
        companion = codex_root / relative
        if not companion.is_file() or sha256_file(companion) != expected:
            raise ValueError(f"Codex companion identity mismatch: {relative}")


def source_file_hashes(source_root: Path) -> dict[str, str]:
    """Hash source payload files, excluding known Python build artifacts."""
    files = {}
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        if (
            relative.parts[0] == "build"
            or any(
                part == "__pycache__" or part.endswith(".egg-info")
                for part in relative.parts
            )
            or relative.as_posix() == ".submission-source-commit"
            or path.suffix in {".pyc", ".pyo"}
            or not path.is_file()
        ):
            continue
        if not path.resolve().is_relative_to(source_root.resolve()):
            raise ValueError(
                f"RoboRSI source symlink escapes its source directory: {relative}"
            )
        files[relative.as_posix()] = sha256_file(path)
    return files


def validate_source_bundle(
    metadata: dict[str, Any], policy_dir: Path = POLICY_DIR
) -> dict[str, str]:
    """Verify the reviewable source bundled in the PR without another checkout."""
    source = metadata["source"]
    bundle = policy_dir / relative_artifact_path(
        source.get("bundle_path"), "source.bundle_path"
    )
    manifest_path = policy_dir / relative_artifact_path(
        source.get("manifest_path"), "source.manifest_path"
    )
    expected_digest = source.get("manifest_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_digest)):
        raise ValueError("submission.json: source.manifest_sha256 is required")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_digest:
        raise ValueError("Committed RoboRSI source manifest is missing or changed")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_commit") != source["commit"]
        or manifest.get("source_url") != source["url"]
    ):
        raise ValueError("Committed RoboRSI source manifest has another revision")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Committed RoboRSI source manifest has no payload files")
    for name, digest in expected.items():
        relative_artifact_path(name, "source manifest file")
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError(f"Invalid committed source digest: {name}")
        if not any(
            name == path or name.startswith(path + "/") for path in SOURCE_PATHS
        ):
            raise ValueError(f"File outside the RoboDojo source export: {name}")
    if bundle.is_symlink() or any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("Committed RoboRSI source bundle must not contain symlinks")
    actual = source_file_hashes(bundle)
    if actual != expected:
        changed = sorted(
            name
            for name in set(expected) | set(actual)
            if expected.get(name) != actual.get(name)
        )
        raise ValueError(
            "Committed RoboRSI source bundle is missing or changed: "
            + ", ".join(changed[:5])
        )
    return expected


def write_source_manifest(
    metadata: dict[str, Any],
    policy_dir: Path,
    source_root: Path,
    *,
    artifact_sha256: str,
    origin: str,
) -> None:
    """Record files directly after copying the committed source bundle."""
    files = source_file_hashes(source_root)
    if not files:
        raise ValueError("The verified RoboRSI source artifact is empty")
    manifest = {
        "schema_version": 1,
        "source_commit": metadata["source"]["commit"],
        "origin": origin,
        "artifact_sha256": artifact_sha256,
        "files": files,
    }
    path = policy_dir / "runtime/source-files.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_source_manifest(
    metadata: dict[str, Any], policy_dir: Path = POLICY_DIR
) -> None:
    """Reject changed or additional source payload files on restart and preflight."""
    source_root = policy_dir / "runtime/RoboRSI"
    manifest_path = policy_dir / "runtime/source-files.json"
    if not manifest_path.is_file():
        raise ValueError(
            "RoboRSI source manifest is missing; reinstall the authenticated source artifact"
        )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_commit") != metadata["source"]["commit"]
    ):
        raise ValueError("RoboRSI source manifest does not match the pinned commit")
    if manifest.get("origin") != "committed_bundle" or manifest.get(
        "artifact_sha256"
    ) != metadata["source"].get("manifest_sha256"):
        raise ValueError(
            "Installed RoboRSI source does not match the committed bundle; "
            "move runtime/RoboRSI aside and reinstall"
        )
    expected = manifest.get("files")
    if expected != validate_source_bundle(metadata, policy_dir):
        raise ValueError("Installed source receipt differs from the committed bundle")
    actual = source_file_hashes(source_root)
    changed = sorted(
        path
        for path in set(expected) | set(actual)
        if expected.get(path) != actual.get(path)
    )
    if changed:
        raise ValueError(
            f"RoboRSI source differs from the verified artifact: {', '.join(changed[:5])}"
        )


def check_runtime(
    config_path: Path,
    *,
    require_key: bool,
    installation_only: bool = False,
    task_name: str | None = None,
    verify_icl_task: str | None = None,
) -> dict[str, Any]:
    """Use the same AgentConfig and executable preflight as the production Model."""
    import yaml
    from roborsi.planner.codex_cli import preflight_codex_cli
    from roborsi.planner.codex_cli_config import CodexCliConfig

    try:
        from .runtime_config import load_agent_config
    except ImportError:
        from runtime_config import load_agent_config

    model_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(model_cfg, dict):
        raise ValueError("deploy config must contain a mapping")
    agent_config = load_agent_config(
        model_cfg, require_key=require_key, installation_only=installation_only
    )
    cli = CodexCliConfig.from_dict(agent_config.document["agent"]["codex_cli"])
    environment = None
    if not require_key:
        # Keep CODEX_API_KEY in the production config while allowing an offline
        # version check. Never mutate os.environ or synthesize credentials.
        environment = dict(os.environ) if cli.inherit_env else {}
        environment.pop("CODEX_API_KEY", None)
        for name in cli.env_vars:
            if name == "CODEX_API_KEY":
                continue
            if name not in os.environ:
                raise ValueError(
                    f"required Codex environment variable is missing: {name}"
                )
            environment[name] = os.environ[name]
        environment.update(cli.env)
    identity = preflight_codex_cli(cli, repo_root=POLICY_DIR, env=environment)
    metadata = load_submission()
    validate_source_manifest(metadata)
    if identity["sha256"] != metadata["cli"]["sha256"]:
        raise ValueError(
            "Installed CLI SHA-256 does not match the confirmed standalone release"
        )
    validate_companions(metadata, POLICY_DIR / "runtime/codex")
    try:
        from .runtime_capabilities import (
            check_python_capabilities,
            check_sandbox_capabilities,
        )
    except ImportError:
        from runtime_capabilities import (
            check_python_capabilities,
            check_sandbox_capabilities,
        )

    capabilities = {
        "python": check_python_capabilities(),
        "sandbox": check_sandbox_capabilities(POLICY_DIR),
    }
    try:
        from .icl_assets import validate_icl_install
    except ImportError:
        from icl_assets import validate_icl_install

    icl = validate_icl_install(POLICY_DIR)
    try:
        from .icl_context import check_icl_materialization, check_icl_selection
    except ImportError:
        from icl_context import check_icl_materialization, check_icl_selection

    if task_name is not None:
        icl = {**icl, "selection": check_icl_selection(POLICY_DIR, task_name)}
    if verify_icl_task is not None:
        icl = {
            **icl,
            "materialization": check_icl_materialization(POLICY_DIR, verify_icl_task),
        }
    try:
        from .pi05_runtime import check_pi05_runtime
        from .pi05_source import load_openpi_client
    except ImportError:
        from pi05_runtime import check_pi05_runtime
        from pi05_source import load_openpi_client

    load_openpi_client(POLICY_DIR)
    if model_cfg.get("vla_backend") == "simate_beta":
        try:
            from .simate_runtime import check_simate_runtime
        except ImportError:
            from simate_runtime import check_simate_runtime
        checkpoint = Path(model_cfg["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = POLICY_DIR / checkpoint
        vla = check_simate_runtime(POLICY_DIR, checkpoint)
    else:
        vla = check_pi05_runtime()
    provider = cli.config_overrides
    # Deliberately print only an allowlist; never serialize AgentConfig or the environment.
    return {
        "source_commit": metadata["source"]["commit"],
        "agent_label": metadata["agent_label"],
        "model": None if installation_only else agent_config.document["agent"]["model"],
        "base_url": None if installation_only else provider["model_providers.runtime.base_url"],
        "deployment_configured": not installation_only,
        "wire_api": provider["model_providers.runtime.wire_api"],
        "executable": identity["executable"],
        "version": identity["version"],
        "sha256": identity["sha256"],
        "source_files": "verified against the authenticated source artifact manifest",
        "check": ("offline installation checks; deployment settings and model access were not tested"
                  if installation_only else "offline CLI identity and configuration; model access was not tested"),
        "key_required": require_key,
        "runtime_capabilities": capabilities,
        "icl": icl,
        **({"vla": vla} if model_cfg.get("vla_backend") == "simate_beta" else {"pi05": vla}),
    }


def main() -> int:
    """Run an offline preflight and report actionable errors without traceback noise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=POLICY_DIR / "deploy.yml")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="check source/CLI installation metadata without importing policy dependencies",
    )
    parser.add_argument(
        "--allow-missing-key",
        action="store_true",
        help="permit only this offline configuration/version check without CODEX_API_KEY",
    )
    parser.add_argument(
        "--installation-only", action="store_true",
        help="validate installed artifacts and CLI offline without deployment settings or credentials",
    )
    parser.add_argument(
        "--task-name",
        help="check the actual evaluation task's pinned ICL selection without copying it",
    )
    parser.add_argument(
        "--verify-icl-task",
        metavar="TASK",
        help="also materialize and read one task's complete ICL package offline",
    )
    args = parser.parse_args()
    try:
        if args.metadata_only and (args.task_name or args.verify_icl_task):
            raise ValueError(
                "ICL selection/materialization checks require installed dependencies"
            )
        metadata = load_submission()
        try:
            from .icl_assets import load_icl_metadata
        except ImportError:
            from icl_assets import load_icl_metadata

        load_icl_metadata(POLICY_DIR)
        try:
            from .pi05_assets import load_pi05_metadata
            from .pi05_source import validate_pi05_source
            from .pi05_tokenizer import load_pi05_tokenizer_metadata
        except ImportError:
            from pi05_assets import load_pi05_metadata
            from pi05_source import validate_pi05_source
            from pi05_tokenizer import load_pi05_tokenizer_metadata

        load_pi05_metadata(POLICY_DIR)
        validate_pi05_source(POLICY_DIR)
        load_pi05_tokenizer_metadata(POLICY_DIR)
        try:
            from .tool_runtime import load_tool_metadata
        except ImportError:
            from tool_runtime import load_tool_metadata

        tools = load_tool_metadata(POLICY_DIR)
        validate_archive(metadata)
        if args.metadata_only:
            result = {
                "source_commit": metadata["source"]["commit"],
                "agent_label": metadata["agent_label"],
                "version": metadata["cli"]["expected_version"],
                "check": "installation metadata only; CLI/model were not executed",
                "tool_python": tools["python_version"],
            }
        else:
            result = check_runtime(
                args.config.resolve(),
                require_key=not (args.allow_missing_key or args.installation_only),
                installation_only=args.installation_only,
                task_name=args.task_name,
                verify_icl_task=args.verify_icl_task,
            )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"RoboRSI preflight failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
