import csv
import json

import numpy as np
import torch

from evals.common.artifacts import EvaluationArtifactWriter
from evals.common.environment import SimulatorEvaluationEnvironment
from evals.common.evaluator import EvaluationConfig, EvaluationRunner
from evals.common.suite import EvaluationTask, TaskSuite
from evals.libero.adapter import LiberoAdapter
from worldscape_policy.types import WorldActionOutput


class FakeRuntime:
    def __init__(self):
        self.policy = torch.nn.Linear(1, 1)
        self.pending = None
        self.commits = 0
        self.discards = 0

    @property
    def has_pending_prediction(self):
        return self.pending is not None

    def reset(self, mode):
        self.mode = mode
        self.pending = None

    def predict(self, **kwargs):
        del kwargs
        self.pending = WorldActionOutput(action=torch.zeros(1, 7))
        return self.pending

    def commit(self, output=None):
        assert output is self.pending
        self.commits += 1
        self.pending = None

    def discard(self):
        self.discards += 1
        self.pending = None


class FakeSimulator:
    def __init__(self, fail=False):
        self.fail = fail
        self.steps = 0
        self.closed = False

    def reset(self, seed=None):
        self.seed = seed
        self.steps = 0
        return _observation(), {}

    def step(self, action):
        assert action.shape == (7,)
        if self.fail:
            raise RuntimeError("rejected")
        self.steps += 1
        return _observation(), 1.0, self.steps == 1, False, {"success": True}

    def close(self):
        self.closed = True


def _observation():
    return {
        "agentview_image": np.zeros((4, 5, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 5, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def test_fake_simulator_suite_writes_common_artifacts(tmp_path):
    native = FakeSimulator()
    runtime = FakeRuntime()
    result = EvaluationRunner(
        runtime,
        SimulatorEvaluationEnvironment(native),
        LiberoAdapter(),
    ).run(
        TaskSuite(
            [
                EvaluationTask("pick", "pick the block"),
                EvaluationTask("place", "place the block"),
            ],
            seed=9,
            suite_id="fake-suite",
            metadata={"setting": "clean"},
        ),
        EvaluationConfig("interactive", max_steps=3, control_frequency_hz=20),
        generator=torch.Generator(),
    )

    summary = EvaluationArtifactWriter(tmp_path).write(
        {"backend": "libero"}, result
    )

    assert runtime.commits == 2
    assert runtime.discards == 0
    assert native.closed
    assert summary["success_rate"] == 1.0
    assert summary["artifact_schema"] == "worldscape-evaluation"
    assert summary["metrics"]["success"]["mean"] == 1.0
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "summary.json").read_text())["successes"] == 2
    episode = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
    assert episode["horizon"] == 3
    assert episode["control_frequency_hz"] == 20
    assert episode["suite_id"] == "fake-suite"
    assert episode["suite_metadata"] == {"setting": "clean"}
    with (tmp_path / "per_task.csv").open() as stream:
        assert {row["task_id"] for row in csv.DictReader(stream)} == {
            "pick",
            "place",
        }
    assert (tmp_path / "config.yaml").is_file()
    assert not (tmp_path / "videos").exists()


def test_fake_simulator_rejection_discards_candidate_state():
    native = FakeSimulator(fail=True)
    runtime = FakeRuntime()
    result = EvaluationRunner(
        runtime,
        SimulatorEvaluationEnvironment(native),
        LiberoAdapter(),
    ).run(
        TaskSuite([EvaluationTask("pick", "pick")]),
        EvaluationConfig("interactive", max_steps=1),
        generator=torch.Generator(),
    )

    assert result.episodes[0].record.status == "failed"
    assert runtime.commits == 0
    assert runtime.discards == 1
    assert not runtime.has_pending_prediction
