from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from worldscape_policy.wam.protocol import WAMPlugin
from worldscape_policy.wam.registry import (
    DEFAULT_WAM_REGISTRY,
    WAMRegistry,
    create_default_wam_registry,
)
from worldscape_policy.wam.wan21 import Wan21WAMConfig
from worldscape_policy.wam.wan22 import Wan22WAMConfig


def test_builtin_plugins_have_versioned_capability_metadata():
    metadata = {
        item.name: item for item in create_default_wam_registry().metadata()
    }

    assert metadata["wan22"].version == "2.2"
    assert {"training", "sampling"} <= metadata["wan22"].capabilities
    assert metadata["wan21"].version == "2.1"
    assert "sampling" not in metadata["wan21"].capabilities


def test_registry_is_closed_over_explicit_config_types():
    registry = WAMRegistry()

    with pytest.raises(TypeError, match="Unregistered WAM config type"):
        registry.construct(object())
    with pytest.raises(KeyError, match="Unknown WAM plugin"):
        registry.get("unknown")


def test_registry_rejects_duplicate_plugin_and_config_registration():
    registry = WAMRegistry()

    def factory(*, config):
        del config

    registry.register(
        name="test",
        version="1",
        capabilities=(),
        config_type=Wan21WAMConfig,
        factory=factory,
    )

    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            name="test",
            version="2",
            capabilities=(),
            config_type=Wan22WAMConfig,
            factory=factory,
        )


def test_builtin_registry_is_sealed_against_runtime_replacement():
    with pytest.raises(RuntimeError, match="sealed"):
        DEFAULT_WAM_REGISTRY.register(
            name="replacement",
            version="1",
            capabilities=(),
            config_type=object,
            factory=lambda *, config: config,
        )


def test_wan21_adapter_is_protocol_complete_but_sampling_is_unsupported():
    plugin = DEFAULT_WAM_REGISTRY.construct(Wan21WAMConfig())
    assert isinstance(plugin, WAMPlugin)

    with pytest.raises(NotImplementedError, match="Wan2.1 sampling is not implemented"):
        plugin.sample(
            reference_frame=torch.zeros(1, 1, 3, 2, 2),
            chunk_latents=torch.zeros(1, 1, 1, 1, 1),
            observation_num_frames=1,
            prompt_signature=("prompt",),
            state=torch.zeros(1, 1, 1),
            embodiment_id=torch.zeros(1, dtype=torch.long),
            cross_attention_tokens=torch.zeros(1, 1, 1),
            negative_cross_attention_tokens=None,
            visual_memory=None,
            generator=torch.Generator(),
        )

    with pytest.raises(NotImplementedError, match="does not support capabilities"):
        DEFAULT_WAM_REGISTRY.construct(
            Wan21WAMConfig(),
            required_capabilities={"sampling"},
        )


def test_wam_model_packages_have_no_platform_branches():
    package = Path(__file__).parents[2] / "src" / "worldscape_policy" / "wam"
    forbidden = {("sys", "platform"), ("os", "name"), ("platform", "system")}
    for source_path in package.rglob("*.py"):
        tree = ast.parse(source_path.read_text())
        for branch in (
            node for node in ast.walk(tree) if isinstance(node, (ast.If, ast.IfExp))
        ):
            attributes = {
                (node.value.id, node.attr)
                for node in ast.walk(branch.test)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
            }
            assert not attributes & forbidden, source_path
