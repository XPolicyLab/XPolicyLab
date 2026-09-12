import json
import sys

import numpy as np
import pytest

from worldscape_policy.cli import evaluate as cli
from evals.agilex.evaluate import main as agilex_main
from evals.agilex.evaluate import run_agilex_recipe
from evals.agilex.robot import (
    AgileXActionCommand,
    AgileXReadRequest,
    LegacyAgileXRobotAdapter,
)


class _LegacyRobot:
    def __init__(self):
        self.sent = []

    def read_observation(self, **kwargs):
        assert kwargs["action_mode"] == "eef"
        image = np.zeros((1, 3, 4, 3), dtype=np.uint8)
        return image, image, image, {"state.left_pos": np.zeros((1, 3))}, 7

    def send_end_pose_action(self, timestamp, rate, left, right):
        self.sent.append((timestamp, rate, left, right))


def test_legacy_transport_names_are_hidden_by_wsp_adapter():
    legacy = _LegacyRobot()
    robot = LegacyAgileXRobotAdapter(legacy)

    observation = robot.observe(AgileXReadRequest(action_mode="eef"))
    result = robot.execute(
        AgileXActionCommand(7, 24, [[0.0]], [[0.0]], action_mode="eef")
    )

    assert observation.timestamp == 7
    assert result.accepted
    assert legacy.sent == [(7, 24, [[0.0]], [[0.0]])]


def test_worldscape_eval_agilex_defaults_to_dry_run(tmp_path, monkeypatch):
    recipe = tmp_path / "agilex.json"
    recipe.write_text(
        json.dumps(
            {
                "backend": "agilex",
                "checkpoint": "/checkpoint",
                "output_dir": str(tmp_path / "artifacts"),
            }
        )
    )
    called = {}

    def run(config, **kwargs):
        called.update(config=config, **kwargs)
        return {"episodes": 1, "success_rate": 0.0}

    monkeypatch.setattr(
        "evals.agilex.evaluate.run_agilex_recipe", run
    )

    assert cli.main(["--config", str(recipe)]) == 0
    assert called["live_hardware"] is False


def test_recipe_cannot_disable_dry_run_without_live_opt_in(tmp_path):
    with pytest.raises(ValueError, match="--live-hardware"):
        run_agilex_recipe(
            {"dry_run": False, "backend_config": {}},
            checkpoint="/checkpoint",
            output_dir=tmp_path,
        )


def test_native_cli_does_not_import_repository_root_legacy_server(monkeypatch):
    monkeypatch.delitem(sys.modules, "dreamzero_real_robot_server", raising=False)

    with pytest.raises(SystemExit):
        agilex_main(["--legacy", "--model-path", "/legacy"])
    assert "dreamzero_real_robot_server" not in sys.modules
