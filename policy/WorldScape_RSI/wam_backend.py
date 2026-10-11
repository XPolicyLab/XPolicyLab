"""Reuse the adjacent company-dev WAM adapter, including its checkpoint transforms."""
import os
from pathlib import Path
import yaml


def build_config(config):
    root = Path(__file__).resolve().parent.parent / "WorldScape_Policy_2"
    cfg = yaml.safe_load((root / "deploy_robodojo.yml").read_text(encoding="utf-8"))
    for key in ("bench_name", "task_name", "ckpt_name", "env_cfg_type", "seed", "device"):
        if config.get(key) is not None:
            cfg[key] = config[key]
    overrides = config.get("wam") or {}
    allowed = {"checkpoint_path", "device", "worldscape_root"}
    if set(overrides) - allowed:
        raise ValueError("Unsupported WAM overrides: " + str(sorted(set(overrides) - allowed)))
    cfg.update(overrides)
    if os.environ.get("WRSI_WAM_CHECKPOINT"):
        cfg["checkpoint_path"] = os.environ["WRSI_WAM_CHECKPOINT"]
    cfg.update(runtime_profile="robodojo", eval_batch=False, action_type="joint")
    cfg["multi_gpu"] = {"enabled": False}
    return cfg


def create(config):
    from XPolicyLab.policy.WorldScape_Policy_2.model import Model
    return Model(build_config(config))
