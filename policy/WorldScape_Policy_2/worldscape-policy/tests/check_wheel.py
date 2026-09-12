"""Dependency-free release wheel smoke check used by CI."""

from __future__ import annotations

import configparser
import sys
import zipfile
from pathlib import Path


def main(path: str) -> None:
    wheel = Path(path)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        entry_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(entry_name).decode())
        scripts = parser["console_scripts"]
        assert set(scripts) == {
            "wsp-train",
            "wsp-eval",
            "wsp-serve",
        }
        required_suffixes = {
            "share/worldscape-policy/configs/model/wsp2_wan22_5b.yaml",
            "share/worldscape-policy/configs/conditioning/interactive_t5.yaml",
            "share/worldscape-policy/configs/conditioning/auto_qwen3vl.yaml",
            "share/worldscape-policy/configs/memory/event_default.yaml",
            "share/worldscape-policy/configs/memory/visual_prefill_default.yaml",
            "share/worldscape-policy/configs/wam/wan22_5b.yaml",
            "share/worldscape-policy/configs/eval/agilex.yaml",
            "share/worldscape-policy/configs/posttrain/agilex.yaml",
            "share/worldscape-policy/configs/posttrain/common_wan22.yaml",
            "share/worldscape-policy/recipes/posttrain/posttrain_agilex_fold_shirt_text_auto.sh",
            "share/worldscape-policy/configs/pretrain/common_wan22.yaml",
            "share/worldscape-policy/configs/pretrain/mixed_three_mode_wan22.yaml",
            "share/worldscape-policy/configs/pretrain/mixed_three_mode_wan22_stage2.yaml",
            "share/worldscape-policy/recipes/pretrain/pretrain_mixed_three_mode_stage1.sh",
            "share/worldscape-policy/recipes/pretrain/pretrain_mixed_three_mode_stage2.sh",
            "share/worldscape-policy/configs/eval/robotwin2_manager.yaml",
            "share/worldscape-policy/recipes/eval/eval_robotwin2.sh",
            "share/worldscape-policy/recipes/posttrain/posttrain_robotwin2_full.sh",
            "share/worldscape-policy/tools/checkpoint/inspect_checkpoint.py",
            "worldscape_policy/cli/serve.py",
            "worldscape_policy/cli/config_composition.py",
            "evals/agilex/evaluate.py",
            "evals/common/checkpoint_runtime.py",
        }
        missing = {
            suffix
            for suffix in required_suffixes
            if not any(name.endswith(suffix) for name in names)
        }
        assert not missing, f"wheel is missing release files: {sorted(missing)}"
        removed_config_directories = ("share/worldscape-policy/configs/policy/",)
        stale_configs = sorted(
            name
            for name in names
            if any(directory in name for directory in removed_config_directories)
        )
        assert not stale_configs, f"wheel contains superseded configs: {stale_configs}"
        assert not any(
            name.endswith("_posttrain.yaml") for name in names
        ), "wheel contains phase-duplicated model or WAM config"
        assert not any(
            name == "socket_test_optimized_AR.py" for name in names
        ), "native wheel must not contain the source-only legacy server"
        forbidden = (
            "evaluation/",
            "groot/",
            "worldscape_policy/compat/",
            "worldscape_policy/legacy_server/",
            "worldscape_policy/legacy_server.py",
        )
        leaked = sorted(
            name for name in names if name.startswith(forbidden)
        )
        assert not leaked, f"native wheel contains legacy source: {leaked}"
        assert "dreamzero-server" not in scripts


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} WHEEL")
    main(sys.argv[1])
