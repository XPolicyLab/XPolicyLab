"""Discover optional policy and simulator hooks in the active XPolicyLab checkout."""

from functools import lru_cache
from importlib import import_module
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[1]
POLICIES = ADAPTER.parents[1] / "policy"


@lru_cache(maxsize=1)
def policy_extensions():
    return {
        path.parents[1].name: import_module(
            f"XPolicyLab.policy.{path.parents[1].name}.rlinf.extension"
        )
        for path in sorted(POLICIES.glob("*/rlinf/extension.py"))
    }


@lru_cache(maxsize=1)
def simulator_extensions():
    extensions = {}
    for path in sorted((ADAPTER / "envs").glob("*/extension.py")):
        module = import_module(
            f"XPolicyLab.infra.RLinf.envs.{path.parent.name}.extension"
        )
        if module.ENV_KEY in extensions:
            raise ValueError(f"Duplicate RLinf simulator key {module.ENV_KEY}")
        extensions[module.ENV_KEY] = module
    return extensions


def simulator_extension(env):
    """Allow a backend-wide hook or a more specific native task ID."""
    backend = env.get("env_type")
    task_id = (env.get("init_params") or {}).get("id")
    extensions = simulator_extensions()
    return extensions.get((backend, task_id), extensions.get((backend, None)))


def active_model(cfg):
    return cfg.rollout.model if cfg.runner.only_eval else cfg.actor.model


def policy_extension(model):
    return policy_extensions().get((model.get("xpolicylab") or {}).get("policy_name"))
