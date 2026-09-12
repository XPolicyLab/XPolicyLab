from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_metadata() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def test_p2_base_install_excludes_capability_specific_heavy_dependencies():
    project = _project_metadata()
    extras = project["optional-dependencies"]

    assert {"core", "train", "server", "agilex", "robotwin2", "libero"} <= extras.keys()
    base = set(project["dependencies"])
    assert base
    for dependency in (
        "ray[default]==2.47.1",
        "deepspeed",
        "tensorrt",
        "gymnasium",
        "mujoco",
        "flask",
    ):
        assert dependency not in base
    for group in ("core", "train", "server", "agilex", "robotwin2", "libero"):
        assert extras[group], f"{group} must remain an installable capability group"
    serialized = repr(project)
    for private_or_unused in (
        "dreamzero",
        "gear",
        "multi-storage-client",
        "nvidia-modelopt",
        "tensorrt",
    ):
        assert private_or_unused not in serialized
    assert "deepspeed==0.18.9" in extras["train"]
    assert "h5py>=3.10,<4" in extras["agilex"]
    assert "mujoco>=3,<4" in extras["robotwin2"]
    assert "mujoco>=3,<4" in extras["libero"]
    assert "websockets>=14,<16" in extras["server"]
    assert "transformers==4.57.1" in base
    assert "transformers==4.57.1" in extras["core"]
    assert "safetensors==0.7.0" in base
    assert "pillow>=10,<13" in base


def test_only_wsp_console_commands_are_published():
    scripts = _project_metadata()["scripts"]

    assert scripts == {
        "wsp-train": "worldscape_policy.cli.train:main",
        "wsp-eval": "worldscape_policy.cli.evaluate:main",
        "wsp-serve": "worldscape_policy.cli.serve:main",
    }


def test_native_wheel_packages_only_public_namespaces():
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    assert (REPO_ROOT / "src/worldscape_policy/cli/serve.py").is_file()
    package_find = metadata["tool"]["setuptools"]["packages"]["find"]
    includes = package_find["include"]
    assert includes == ["worldscape_policy*", "evals*"]
    assert "exclude" not in package_find


def test_release_documents_are_present():
    for relative_path in (
        "LICENSE",
        "MODEL_CARD.md",
        "docs/provenance.md",
        "docs/architecture.md",
        "docs/evaluation.md",
        "docs/posttraining.md",
        "requirements.txt",
        "environment.yml",
    ):
        assert (REPO_ROOT / relative_path).is_file()


def test_public_dependency_files_do_not_reference_private_packages_or_paths():
    paths = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "environment.yml",
        *REPO_ROOT.glob("requirements*.txt"),
    ]
    forbidden = (
        "/private/" + "internal-project",
        "/private/" + "internal-storage",
        "dreamzero",
        "multi-storage-client",
        "nvidia-modelopt",
        "tensorrt",
    )
    for path in paths:
        text = path.read_text()
        assert not any(token in text for token in forbidden), path.name
