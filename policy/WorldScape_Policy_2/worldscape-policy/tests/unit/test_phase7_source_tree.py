from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_SOURCE_TOKENS = (
    "import " + "groot",
    "from " + "groot",
    "import " + "dreamzero",
    "from " + "dreamzero",
    "worldscape_policy" + ".compat",
    "/private/" + "internal-project",
)


def test_native_source_boundary_has_no_legacy_imports_or_internal_paths():
    offenders: list[str] = []
    roots = (REPO_ROOT / "src" / "worldscape_policy", REPO_ROOT / "tests" / "unit")
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text()
            if any(token in text for token in FORBIDDEN_SOURCE_TOKENS):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []


def test_legacy_runtime_trees_are_physically_absent():
    for relative in (
        "groot",
        "scripts",
        "src/worldscape_policy/compat",
        "src/worldscape_policy/legacy_server",
        "src/worldscape_policy/evaluation",
        "src/worldscape_policy/runtime",
    ):
        assert not (REPO_ROOT / relative).exists()


def test_only_public_console_scripts_are_published():
    metadata = (REPO_ROOT / "pyproject.toml").read_text()
    for command in ("wsp-train", "wsp-eval", "wsp-serve"):
        assert f"{command} =" in metadata
    assert "worldscape-train =" not in metadata
