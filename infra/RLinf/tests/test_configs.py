"""Compose shipped configs without starting Ray, CUDA, or a model."""

import json
import os
from pathlib import Path
import subprocess
from unittest.mock import Mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from XPolicyLab.infra.RLinf.core.configuration import config_dirs, register_config_paths
from XPolicyLab.infra.RLinf.core.validation import validate_adapter_config

register_config_paths()
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    p.stem
    for directory in config_dirs()
    for p in directory.glob("*.yaml")
    if not p.stem.endswith("_sft")
]


@pytest.mark.parametrize("suffix", ["ppo_xpolicylab_pi05", "grpo_xpolicylab_pi05", "xpolicylab_pi05_eval"])
def test_robodojo_task_override(suffix, monkeypatch):
    monkeypatch.setenv("RLINF_PATH", str(ROOT / "RLinf"))
    monkeypatch.delenv("PI05_REPO_ID", raising=False)
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(config_name=f"robodojo_stack_bowls_{suffix}",
                      overrides=["task_name=align_blocks", "episode_steps=200"])
    for mode in (["eval"] if cfg.runner.only_eval else ["train", "eval"]):
        env = cfg.env[mode]
        assert env.init_params.task_name == "align_blocks"
        assert env.max_steps_per_rollout_epoch == env.max_episode_steps == 200
    model = cfg.rollout.model if cfg.runner.only_eval else cfg.actor.model
    assert model.xpolicylab.task_name == "align_blocks"
    assert model.xpolicylab.repo_id == "robodojo_align_blocks"
    assert "align_blocks" in cfg.runner.logger.experiment_name
    monkeypatch.setenv("PI05_REPO_ID", "shared_checkpoint_stats")
    assert model.xpolicylab.repo_id == "shared_checkpoint_stats"


@pytest.mark.parametrize("name", CONFIGS)
def test_compose(name, monkeypatch):
    for key in (
        "POLICY_MODEL_PATH",
        "ROBOTWIN_ASSETS_PATH",
        "ROBODOJO_ROOT",
        "ROBODOJO_SIM_PYTHON",
        "ROBOTWIN_QUATERNION_ORDER",
    ):
        monkeypatch.setenv(key, "/not-installed")
    monkeypatch.setenv("RLINF_PATH", str(ROOT / "RLinf"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(config_name=name)
    OmegaConf.to_container(cfg, resolve=True)
    from rlinf import config as native
    from rlinf.scheduler.cluster.utils import parse_rank_config
    from XPolicyLab.infra.RLinf.register import register

    # Resolve "all" against a simulated eight-GPU node without starting Ray.
    sizes = {
        component.strip(): len(parse_rank_config(ranks, available_ranks=list(range(8))))
        for components, ranks in cfg.cluster.component_placement.items()
        for component in components.split(",")
    }
    placement = Mock()
    placement.get_world_size.side_effect = sizes.__getitem__
    monkeypatch.setattr(native, "Cluster", Mock())
    monkeypatch.setattr(native, "HybridComponentPlacement", Mock(return_value=placement))
    register()
    validate_adapter_config(cfg)
    if "pi05" in name and not cfg.runner.only_eval:
        cfg.algorithm.recompute_logprobs = True
    native.validate_cfg(cfg)
    model = cfg.rollout.model if cfg.runner.only_eval else cfg.actor.model
    if "pi05" in name:
        assert model.xpolicylab.rl_backend == "rlinf_v0_4"
        assert cfg.env.eval.total_num_envs == 128
        if not cfg.runner.only_eval:
            assert cfg.actor.fsdp_config.sharding_strategy == "full_shard"
            ppo = cfg.algorithm.adv_type == "gae"
            train = cfg.env.train
            assert (train.rollout_epoch, train.total_num_envs, train.group_size) == (
                (4, 256, 1) if ppo else (8, 64, 8)
            )
            transitions = train.rollout_epoch * train.total_num_envs * (
                train.max_steps_per_rollout_epoch // model.num_action_chunks
            )
            assert transitions % cfg.actor.global_batch_size == 0
            assert (cfg.runner.max_epochs, cfg.runner.val_check_interval) == (1000, 10)
            assert cfg.algorithm.update_epoch == (5 if ppo else 1)
        assert model.model_type == "openpi"


@pytest.mark.parametrize("sharding", ["no_shard", "shard_grad_op", "full_shard"])
def test_pi05_uses_native_fsdp_validation(monkeypatch, sharding):
    from rlinf.config import validate_fsdp_cfg

    monkeypatch.setenv("RLINF_PATH", str(ROOT / "RLinf"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(config_name="robotwin_adjust_bottle_ppo_xpolicylab_pi05")
    cfg.actor.fsdp_config.sharding_strategy = sharding
    cfg.actor.fsdp_config.use_orig_params = False
    assert validate_fsdp_cfg(cfg.actor).fsdp_config.use_orig_params
    cfg.actor.fsdp_config.mixed_precision.param_dtype = "bf16"
    with pytest.raises(ValueError, match="mixed_precision_rules"):
        validate_fsdp_cfg(cfg.actor)


@pytest.mark.parametrize("mode,batches", [("train", 4), ("eval", 2)])
def test_robotwin_consumes_all_official_seeds(mode, batches, monkeypatch):
    from collections import Counter
    from XPolicyLab.infra.RLinf.core.extensions import simulator_extension

    monkeypatch.setenv("RLINF_PATH", str(ROOT / "RLinf"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(config_name="robotwin_adjust_bottle_ppo_xpolicylab_pi05").env[mode]
    seeds = json.loads(Path(cfg.seeds_path).read_text())[cfg.task_config.task_name]["success_seeds"]
    cls = simulator_extension(cfg).env_class(cfg)
    monkeypatch.setattr(cls, "_init_env", lambda self: None)
    ranks = 8
    workers = [cls(cfg, cfg.total_num_envs // ranks, rank, ranks, None) for rank in range(ranks)]
    seen = Counter()
    for _ in range(batches):
        batch = [s for worker in workers for s in worker.reset_state_ids.tolist()]
        assert len(set(batch)) == cfg.total_num_envs
        seen.update(batch)
        for worker in workers:
            worker.update_reset_state_ids(list(range(worker.num_envs)) if mode == "eval" else None)
    assert set(seen) == set(seeds)
    assert max(seen.values()) - min(seen.values()) <= 1


@pytest.mark.parametrize(
    "override",
    [
        "env.train.auto_reset=true",
        "env.enable_offload=true",
        "env.train.max_episode_steps=799",
        "env.train.group_size=3",
        "algorithm.reward_type=action_level",
        "actor.model.xpolicylab.action_type=ee",
    ],
)
def test_robodojo_rejects_incompatible_runtime(override, monkeypatch):
    monkeypatch.setenv("RLINF_PATH", str(ROOT / "RLinf"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(
            config_name="robodojo_stack_bowls_ppo_xpolicylab_pi05", overrides=[override]
        )
    with pytest.raises(ValueError):
        validate_adapter_config(cfg)


@pytest.mark.parametrize(
    "script,args,address,entry",
    [
        ("run.sh", [], "local", "run.py"),
        ("run.sh", ["robotwin_adjust_bottle_ppo_xpolicylab_pi05"], "local", "run.py"),
        ("run.sh", ["robotwin_adjust_bottle_ppo_xpolicylab_pi05", "--help"], "", "check_config.py"),
        ("run.sh", ["robotwin_adjust_bottle_xpolicylab_pi05_eval", "--cfg", "job"], "", "check_config.py"),
    ],
)
def test_launcher_skips_ray_start(tmp_path, monkeypatch, script, args, address, entry):
    launcher = tmp_path / "XPolicyLab" / "infra" / "RLinf" / "scripts" / script
    launcher.parent.mkdir(parents=True)
    for name in {"run.sh", script}:
        (launcher.parent / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    for key in ("POLICY_MODEL_PATH", "ROBOTWIN_ASSETS_PATH", "ROBOTWIN_RUNTIME_PATH"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("ROBOTWIN_QUATERNION_ORDER", "wxyz")
    ray = tmp_path / "ray"
    ray.write_text('#!/bin/sh\necho "unexpected Ray call: $*" >&2\nexit 7\n')
    ray.chmod(0o755)
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\n[ "$1" = "-c" ] || echo "${1##*/} $RAY_ADDRESS"\n')
    python.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("RAY_ADDRESS", address)
    launch = subprocess.run(["bash", str(launcher), *args], capture_output=True, text=True)
    assert launch.returncode == 0, launch.stderr
    assert launch.stdout.split() == [entry] + ([address] if address else [])
