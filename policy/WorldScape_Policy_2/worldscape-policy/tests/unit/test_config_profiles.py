from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from worldscape_policy.cli.config_composition import (
    ConfigCompositionError,
    load_composed_config,
)
from worldscape_policy.cli.config_profiles import resolve_config_profiles


def _config():
    return OmegaConf.create(
        {
            "selectors": {
                "mode": "interactive",
                "visual_prompt": "demo",
                "dataset_name": "demo_data",
            },
            "profile_order": ["mode", "visual_prompt", "dataset_name"],
            "profiles": {
                "mode": {
                    "interactive": {"model": {"mode": "interactive", "width": 1}},
                    "auto": {"model": {"mode": "auto", "width": 2}},
                },
                "visual_prompt": {
                    "none": {"data": {"sampling": "none"}},
                    "demo": {"data": {"sampling": "uniform"}},
                },
                "dataset_name": {
                    "demo_data": {
                        "_requires_": {
                            "mode": "interactive",
                            "visual_prompt": ["demo"],
                        },
                        "data": {"name": "demo_data"},
                    }
                },
            },
            "model": {"width": 0},
            "data": {},
        }
    )


def test_profiles_merge_in_declared_order_and_cli_overrides_win():
    result = resolve_config_profiles(
        _config(),
        overrides=OmegaConf.from_dotlist(["model.width=9"]),
    )

    assert result.model.mode == "interactive"
    assert result.model.width == 9
    assert result.data == {"sampling": "uniform", "name": "demo_data"}
    assert "profiles" not in result
    assert result.selectors.visual_prompt == "demo"


def test_selector_override_changes_selected_profile():
    config = _config()
    config.profiles.dataset_name.demo_data._requires_.mode = [
        "interactive",
        "auto",
    ]
    result = resolve_config_profiles(
        config,
        overrides=OmegaConf.from_dotlist(["selectors.mode=auto"]),
    )

    assert result.model.mode == "auto"
    assert result.selectors.mode == "auto"


def test_profile_rejects_incompatible_selectors():
    config = _config()
    config.selectors.visual_prompt = "none"

    with pytest.raises(ValueError, match="requires visual_prompt"):
        resolve_config_profiles(config)


def test_profile_rejects_unknown_selector():
    config = _config()
    config.selectors.mode = "missing"

    with pytest.raises(ValueError, match="Unsupported mode selector"):
        resolve_config_profiles(config)


def test_plain_config_is_unchanged_except_overrides():
    config = OmegaConf.create({"value": 1})

    result = resolve_config_profiles(
        config,
        overrides=OmegaConf.from_dotlist(["value=2"]),
    )

    assert result.value == 2


def test_recursive_relative_includes_merge_in_order_then_overlay(tmp_path):
    fragments = tmp_path / "fragments"
    fragments.mkdir()
    (fragments / "base.yaml").write_text(
        "value: base\nnested:\n  base: true\n  winner: base\n"
    )
    (fragments / "middle.yaml").write_text(
        "includes: [base.yaml]\nnested:\n  middle: true\n  winner: middle\n"
    )
    (tmp_path / "last.yaml").write_text(
        "nested:\n  last: true\n  winner: last\n"
    )
    root = tmp_path / "root.yaml"
    root.write_text(
        "includes:\n"
        "  - fragments/middle.yaml\n"
        "  - last.yaml\n"
        "nested:\n"
        "  winner: root\n"
    )

    result = load_composed_config(root)

    assert OmegaConf.to_container(result) == {
        "value": "base",
        "nested": {
            "base": True,
            "middle": True,
            "last": True,
            "winner": "root",
        },
    }
    assert "includes" not in result


def test_duplicate_include_rejects_resolved_path_aliases(tmp_path):
    (tmp_path / "shared.yaml").write_text("value: 1\n")
    root = tmp_path / "root.yaml"
    root.write_text("includes: [shared.yaml, ./shared.yaml]\n")

    with pytest.raises(ConfigCompositionError, match="Duplicate config include"):
        load_composed_config(root)


def test_include_cycle_reports_actionable_chain(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("includes: [second.yaml]\n")
    second.write_text("includes: [first.yaml]\n")

    with pytest.raises(ConfigCompositionError, match=r"first\.yaml.*second\.yaml.*first\.yaml"):
        load_composed_config(first)


def test_missing_include_is_rejected_with_declaring_file(tmp_path):
    root = tmp_path / "root.yaml"
    root.write_text("includes: [missing.yaml]\n")

    with pytest.raises(FileNotFoundError, match=r"missing\.yaml.*included by"):
        load_composed_config(root)


@pytest.mark.parametrize(
    ("name", "contents", "message"),
    [
        ("root-list.yaml", "- item\n", "root must be a mapping"),
        ("scalar-includes.yaml", "includes: child.yaml\n", "must be a list"),
        ("mapping-includes.yaml", "includes: {child: true}\n", "must be a list"),
        ("invalid-entry.yaml", "includes: [1]\n", "non-empty path string"),
    ],
)
def test_invalid_roots_and_includes_are_rejected(
    tmp_path, name, contents, message
):
    path = tmp_path / name
    path.write_text(contents)

    with pytest.raises(ConfigCompositionError, match=message):
        load_composed_config(path)


def test_composition_profiles_then_cli_overrides_preserve_priority(tmp_path):
    (tmp_path / "profiles.yaml").write_text(
        "profile_order: [mode]\n"
        "selectors: {mode: interactive}\n"
        "profiles:\n"
        "  mode:\n"
        "    interactive: {model: {mode: interactive, width: 1}}\n"
        "    auto: {model: {mode: auto, width: 2}}\n"
    )
    root = tmp_path / "root.yaml"
    root.write_text("includes: [profiles.yaml]\nmodel: {source: root}\n")
    overrides = OmegaConf.from_dotlist(
        ["selectors.mode=auto", "model.width=9"]
    )

    result = resolve_config_profiles(
        load_composed_config(root),
        overrides=overrides,
    )

    assert result.selectors.mode == "auto"
    assert result.model == {"mode": "auto", "width": 9, "source": "root"}
