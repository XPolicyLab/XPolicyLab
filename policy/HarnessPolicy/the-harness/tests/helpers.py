"""Configurations for tests: the agent of configs/ (or one of its ablations) with a model
file, on a deploy where every service runs on its default port
(tests/fixtures/deploy-test.toml)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

TEST_DEPLOY = ROOT / "tests" / "fixtures" / "deploy-test.toml"


def agent_file(name="default") -> Path:
    """configs/default.toml for "default", else configs/ablations/<name>.toml."""
    return ROOT / "configs" / ("default.toml" if name == "default" else f"ablations/{name}.toml")


def load_test_experiment(agent="default", model=None, *, tasks=("general_pickup",), seeds=(0,), sets=(),
                         deploy=TEST_DEPLOY, bench="robodojo", strict=None):
    from harness_policy.config.experiments import load_experiment

    tree = {"agent": str(agent_file(agent)), "tasks": list(tasks), "seeds": list(seeds), "layouts": "procedural",
            "endpoint": "http://127.0.0.1:8771", "bench": bench}
    if model:
        tree["model"] = model
    return load_experiment(tree=tree, deploy=str(deploy), sets=sets, strict=strict)


def task_config(agent="default", model=None, task="general_pickup", **options):
    return load_test_experiment(agent, model, tasks=(task,), **options).task_config(task)
