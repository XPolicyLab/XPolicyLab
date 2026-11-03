"""Exercise real sealed experience loading without model, simulator or large assets."""

from __future__ import annotations

from importlib import import_module

import base64
import hashlib
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

icl_context = import_module("XPolicyLab.policy.SenseNova-RoboRSI.icl_context")
check_icl_materialization = import_module("XPolicyLab.policy.SenseNova-RoboRSI.icl_context").check_icl_materialization
check_icl_selection = import_module("XPolicyLab.policy.SenseNova-RoboRSI.icl_context").check_icl_selection
prepare_icl_context = import_module("XPolicyLab.policy.SenseNova-RoboRSI.icl_context").prepare_icl_context
icl_fixture = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.icl_fixture")
FIXTURE_BUNDLE_ID = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.icl_fixture").FIXTURE_BUNDLE_ID
FIXTURE_EXPERIENCE_TEXT = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.icl_fixture").FIXTURE_EXPERIENCE_TEXT
FIXTURE_IMAGE_BYTES = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.icl_fixture").FIXTURE_IMAGE_BYTES
write_icl_fixture = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.icl_fixture").write_icl_fixture


@pytest.fixture
def policy(tmp_path):
    directory = tmp_path / "policy"
    write_icl_fixture(directory, "stack_bowls")
    return directory


def rewrite(path, update):
    document = json.loads(path.read_text())
    update(document)
    path.write_text(json.dumps(document))


def repin_inputs(policy):
    """Change only a private test policy's explicit metadata after fixture edits."""
    rewrite(
        policy / "submission.json",
        lambda document: document["icl"].update(
            catalog_sha256=hashlib.sha256(
                (policy / "icl/catalog.json").read_bytes()
            ).hexdigest(),
            artifacts_sha256=hashlib.sha256(
                (policy / "icl/artifacts.json").read_bytes()
            ).hexdigest(),
        ),
    )


def test_loads_selected_experience_without_preloading_historical_commands(
    policy, tmp_path
):
    result = prepare_icl_context(policy, "stack_bowls", tmp_path / "episode")
    assert result.workspace == tmp_path / "episode/icl"
    assert result.source_identity["bundle_id"] == FIXTURE_BUNDLE_ID
    assert result.source_identity["mapping"]["kind"] == "original_preparation"
    assert (
        result.context.skills
        == result.context.skill_paths
        == result.context.memory
        == ()
    )
    prompt = "\n".join(result.context.initial_context)
    assert FIXTURE_EXPERIENCE_TEXT.strip() not in prompt
    assert "HISTORICAL_EXPERIENCE_SENTINEL" not in prompt
    assert "roborsi_action once per step" in prompt
    assert "historical interfaces are not available" in prompt
    assert "START_HERE.md" in prompt and "icl_read_text" in prompt
    assert "repeat counts" in prompt and "numeric gripper" in prompt
    assert str(result.workspace) in prompt and "reference_root" in prompt
    assert ".npz/.npy files as pure data" in prompt
    assert "Do not execute\nhistorical robot/VLA control code" in prompt
    assert {tool.name for tool in result.tools} == {
        "icl_list",
        "icl_read_text",
        "icl_read_image",
    }
    assert not (policy / "icl/source-inventory.json").exists()


def test_valid_reference_image_larger_than_one_mib_preserves_all_bytes(
    tmp_path, monkeypatch
):
    output = io.BytesIO()
    Image.new("RGB", (600, 600), (23, 45, 67)).save(
        output, format="PNG", compress_level=0
    )
    image_bytes = output.getvalue()
    assert len(image_bytes) > 1024 * 1024
    monkeypatch.setattr(icl_fixture, "FIXTURE_IMAGE_BYTES", image_bytes)
    policy = tmp_path / "large-image-policy"
    write_icl_fixture(policy, "stack_bowls")
    result = prepare_icl_context(policy, "stack_bowls", tmp_path / "episode")
    image_tool = next(tool for tool in result.tools if tool.name == "icl_read_image")
    response = image_tool.handler({"path": "evidence/reference.png"})
    assert "error" not in response.result
    blocks = [block for block in response.content_blocks if block["type"] == "image"]
    assert len(blocks) == 1
    assert base64.b64decode(blocks[0]["source"]["data"]) == image_bytes
    assert (result.workspace / "evidence/reference.png").read_bytes() == image_bytes


def test_tools_read_episode_copy_and_preserve_evidence_image_bytes(policy, tmp_path):
    result = prepare_icl_context(policy, "stack_bowls", tmp_path / "episode")
    tools = {tool.name: tool for tool in result.tools}
    listing = tools["icl_list"].handler({}).result
    assert {entry["path"] for entry in listing["entries"]} == {
        "START_HERE.md",
        "evidence",
        "skills",
    }
    text = tools["icl_read_text"].handler({"path": "skills/public_demo/SKILL.md"})
    assert text.result["text"] == FIXTURE_EXPERIENCE_TEXT
    image = tools["icl_read_image"].handler({"path": "evidence/reference.png"})
    blocks = [block for block in image.content_blocks if block["type"] == "image"]
    assert len(blocks) == 1
    assert base64.b64decode(blocks[0]["source"]["data"]) == FIXTURE_IMAGE_BYTES
    assert blocks[0]["source"]["media_type"] == "image/png"
    denied = tools["icl_read_text"].handler({"path": "../../submission.json"})
    assert denied.result["retryable"] is True
    assert "error" in denied.result
    before = (result.workspace / "START_HERE.md").read_bytes()
    extra_argument = tools["icl_read_text"].handler(
        {"path": "START_HERE.md", "text": "overwrite"}
    )
    assert "error" in extra_argument.result
    assert (result.workspace / "START_HERE.md").read_bytes() == before


def test_episode_copies_are_independent_and_frozen_source_is_unchanged(
    policy, tmp_path
):
    first = prepare_icl_context(policy, "stack_bowls", tmp_path / "episode-1")
    second = prepare_icl_context(policy, "stack_bowls", tmp_path / "episode-2")
    source = (
        policy / "runtime/icl/packages" / FIXTURE_BUNDLE_ID / "payload/START_HERE.md"
    )
    original = source.read_bytes()
    (first.workspace / "START_HERE.md").write_text("Private episode notes")
    assert source.read_bytes() == original
    assert (second.workspace / "START_HERE.md").read_bytes() == original
    shutil.rmtree(first.workspace.parent)
    assert second.workspace.is_dir() and source.is_file()


def test_unknown_task_and_missing_payload_never_fall_back(policy, tmp_path):
    with pytest.raises(ValueError, match="No pinned ICL experience mapping"):
        prepare_icl_context(policy, "stack_bowls_random", tmp_path / "unknown")
    shutil.rmtree(policy / "runtime/icl/packages" / FIXTURE_BUNDLE_ID)
    with pytest.raises(ValueError, match="payload.*missing.*install.sh"):
        prepare_icl_context(policy, "stack_bowls", tmp_path / "missing")
    assert not (tmp_path / "unknown/icl").exists()


@pytest.mark.parametrize("target", ["manifest", "payload"])
def test_changed_manifest_or_payload_is_rejected(policy, tmp_path, target):
    package = policy / "runtime/icl/packages" / FIXTURE_BUNDLE_ID
    if target == "manifest":
        with (package / "manifest.json").open("a") as stream:
            stream.write("\n")
        error = "manifest hash mismatch"
    else:
        (package / "payload/START_HERE.md").write_text("Changed frozen reference")
        error = "file inventory differs"
    with pytest.raises(ValueError, match=error):
        prepare_icl_context(policy, "stack_bowls", tmp_path / "episode")


def test_explicit_base_task_transfer_is_reported_without_loading_scoring_evidence(
    policy, tmp_path
):
    def transfer(document):
        document["tasks"]["stack_bowls_random"] = FIXTURE_BUNDLE_ID
        document["task_mapping_details"]["stack_bowls_random"] = {
            "kind": "base_task_transfer",
            "base_task": "stack_bowls",
            "source_instances": ["stack_bowls-s1", "stack_bowls-s2"],
            "reason": "Explicit base-task knowledge transfer; no random-specific demonstrations.",
            "evidence": [{"path": "task/scoring_source.py", "sha256": "4" * 64}],
        }

    rewrite(policy / "icl/catalog.json", transfer)
    repin_inputs(policy)
    result = prepare_icl_context(policy, "stack_bowls_random", tmp_path / "episode")
    assert result.source_identity["mapping"]["base_task"] == "stack_bowls"
    assert result.source_identity["mapping"]["kind"] == "base_task_transfer"
    prompt = "\n".join(result.context.initial_context)
    assert "no random-specific demonstrations" in prompt
    assert "scoring_source.py" not in prompt


def test_even_pinned_package_cannot_preload_historical_instructions(policy, tmp_path):
    manifest = policy / "runtime/icl/packages" / FIXTURE_BUNDLE_ID / "manifest.json"
    rewrite(
        manifest,
        lambda document: document.update(skills=["skills/public_demo/SKILL.md"]),
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rewrite(
        policy / "icl/artifacts.json",
        lambda document: document["artifacts"][FIXTURE_BUNDLE_ID].update(
            manifest_sha256=digest
        ),
    )
    repin_inputs(policy)
    with pytest.raises(ValueError, match="must not preload historical instructions"):
        prepare_icl_context(policy, "stack_bowls", tmp_path / "episode")


def test_catalog_payload_counts_and_upstream_identity_are_checked(policy, tmp_path):
    catalog = policy / "icl/catalog.json"
    rewrite(
        catalog,
        lambda document: document["bundles"][FIXTURE_BUNDLE_ID].update(file_count=999),
    )
    repin_inputs(policy)
    with pytest.raises(ValueError, match="payload counts or bytes"):
        prepare_icl_context(policy, "stack_bowls", tmp_path / "bad-count")
    rewrite(catalog, lambda document: document.update(source_commit="0" * 40))
    repin_inputs(policy)
    with pytest.raises(ValueError, match="different upstream identities"):
        prepare_icl_context(policy, "stack_bowls", tmp_path / "bad-source")


@pytest.mark.parametrize("name", ["catalog", "artifacts"])
def test_tracked_icl_locks_match_submission_pins(policy, tmp_path, name):
    with (policy / "icl" / (name + ".json")).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_icl_context(policy, "stack_bowls", tmp_path / "episode")
    assert not (tmp_path / "episode/icl").exists()


@pytest.mark.parametrize("ancestor", ["runtime", "runtime/icl", "runtime/icl/packages"])
def test_shared_runtime_ancestors_materialize_real_independent_copies(
    policy, tmp_path, ancestor
):
    selected = policy / ancestor
    shared = tmp_path / "shared-read-only-contents"
    selected.rename(shared)
    selected.symlink_to(shared, target_is_directory=True)
    result = prepare_icl_context(policy, "stack_bowls", tmp_path / "episode")
    source = policy / "runtime/icl/packages" / FIXTURE_BUNDLE_ID / "payload"
    assert result.source_identity["entry_points"][0] == "START_HERE.md"
    for path in source.rglob("*"):
        if path.is_file():
            copied = result.workspace / path.relative_to(source)
            assert copied.read_bytes() == path.read_bytes()
            assert not copied.is_symlink()
            assert copied.stat().st_ino != path.stat().st_ino
    reader = next(tool for tool in result.tools if tool.name == "icl_read_text")
    assert (
        reader.handler({"path": "START_HERE.md"}).result["text"]
        == (source / "START_HERE.md").read_text()
    )


@pytest.mark.parametrize("linked_runtime", [False, True])
@pytest.mark.parametrize(
    "target", ["package", "payload", "payload-file", "payload-directory"]
)
def test_shared_runtime_does_not_allow_links_inside_the_sealed_package(
    policy, tmp_path, linked_runtime, target
):
    if linked_runtime:
        runtime = policy / "runtime"
        shared = tmp_path / "shared-runtime"
        runtime.rename(shared)
        runtime.symlink_to(shared, target_is_directory=True)
    package = policy / "runtime/icl/packages" / FIXTURE_BUNDLE_ID
    selected = {
        "package": package,
        "payload": package / "payload",
        "payload-file": package / "payload/START_HERE.md",
        "payload-directory": package / "payload/evidence",
    }[target]
    outside = tmp_path / "outside-sealed-package"
    selected.rename(outside)
    selected.symlink_to(outside, target_is_directory=outside.is_dir())
    with pytest.raises(ValueError, match="symlink|cannot be links"):
        prepare_icl_context(policy, "stack_bowls", tmp_path / "episode")
    assert not (tmp_path / "episode/icl").exists()


def source_records(root):
    return {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            path.stat().st_nlink,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def track_temporary_directories(monkeypatch):
    original = icl_context.tempfile.TemporaryDirectory
    created = []

    def tracked(*args, **kwargs):
        temporary = original(*args, **kwargs)
        created.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(icl_context.tempfile, "TemporaryDirectory", tracked)
    return created


def test_install_check_materializes_reads_and_removes_its_private_copy(
    policy, tmp_path, monkeypatch
):
    runtime = policy / "runtime"
    shared = tmp_path / "shared-runtime"
    runtime.rename(shared)
    runtime.symlink_to(shared, target_is_directory=True)
    before = source_records(shared)
    created = track_temporary_directories(monkeypatch)
    receipt = check_icl_materialization(policy, "stack_bowls")
    assert receipt["task_name"] == "stack_bowls"
    assert receipt["bundle_id"] == FIXTURE_BUNDLE_ID
    assert receipt["payload_files"] == 4
    assert receipt["materialization_verified"] is True
    assert receipt["reader_verified"] is True
    assert receipt["reader_characters"] > 0
    assert receipt["temporary_copy_removed"] is True
    assert receipt["model_or_api_used"] is False
    assert created and all(not path.exists() for path in created)
    assert all(not path.is_relative_to(policy) for path in created)
    assert "HISTORICAL_EXPERIENCE_SENTINEL" not in json.dumps(receipt)
    assert source_records(shared) == before


def test_install_check_removes_copy_if_reader_validation_fails(policy, monkeypatch):
    before = source_records(policy / "runtime")
    created = track_temporary_directories(monkeypatch)
    original = icl_context.prepare_icl_context

    def broken_reader(*args, **kwargs):
        prepared = original(*args, **kwargs)
        tools = tuple(
            SimpleNamespace(
                name=tool.name,
                handler=lambda _args: SimpleNamespace(
                    result={"error": "synthetic reader failure"}
                ),
            )
            if tool.name == "icl_read_text"
            else tool
            for tool in prepared.tools
        )
        return replace(prepared, tools=tools)

    monkeypatch.setattr(icl_context, "prepare_icl_context", broken_reader)
    with pytest.raises(ValueError, match="text reader failed"):
        check_icl_materialization(policy, "stack_bowls")
    assert created and all(not path.exists() for path in created)
    assert source_records(policy / "runtime") == before


def test_selection_checks_shared_package_metadata_without_reading_payload(
    policy, tmp_path, monkeypatch
):
    runtime = policy / "runtime"
    shared = tmp_path / "shared-runtime"
    runtime.rename(shared)
    runtime.symlink_to(shared, target_is_directory=True)
    original_hash = icl_context._sha256
    hashed = []

    def metadata_hash(path):
        assert "payload" not in path.parts
        hashed.append(path)
        return original_hash(path)

    monkeypatch.setattr(icl_context, "_sha256", metadata_hash)
    receipt = check_icl_selection(policy, "stack_bowls")
    assert receipt["package_root"] == str(shared / "icl/packages" / FIXTURE_BUNDLE_ID)
    assert receipt["bundle_id"] == FIXTURE_BUNDLE_ID
    assert receipt["payload_hashes_verified"] is False
    assert receipt["materialization_verified"] is False
    assert {path.name for path in hashed} == {
        "catalog.json",
        "artifacts.json",
        "manifest.json",
    }
    assert not (policy / "icl").joinpath("workspace").exists()


@pytest.mark.parametrize("target", ["package", "manifest", "payload", "entry-point"])
def test_light_selection_rejects_linked_package_layout(policy, tmp_path, target):
    package = policy / "runtime/icl/packages" / FIXTURE_BUNDLE_ID
    selected = {
        "package": package,
        "manifest": package / "manifest.json",
        "payload": package / "payload",
        "entry-point": package / "payload/START_HERE.md",
    }[target]
    outside = tmp_path / "outside-sealed-package"
    selected.rename(outside)
    selected.symlink_to(outside, target_is_directory=outside.is_dir())
    with pytest.raises(
        ValueError, match="symlink|cannot be links|manifest hash mismatch"
    ):
        check_icl_selection(policy, "stack_bowls")


def test_light_selection_defers_full_payload_integrity_to_episode_prepare(
    policy, tmp_path
):
    source = policy / "runtime/icl/packages" / FIXTURE_BUNDLE_ID
    (source / "payload/evidence/geometry.json").write_text("Changed source payload")
    receipt = check_icl_selection(policy, "stack_bowls")
    assert receipt["payload_hashes_verified"] is False
    with pytest.raises(ValueError, match="file inventory differs"):
        prepare_icl_context(policy, "stack_bowls", tmp_path / "episode")
