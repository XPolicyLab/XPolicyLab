import json

import numpy as np
import torch

from evals.agilex.evaluate import (
    RealRobotRunner,
    RealRobotRunnerConfig,
)
from evals.agilex.observation_adapter import AgileXObservationAdapter
from evals.common import EvaluationArtifactWriter
from evals.common.checkpoint_runtime import (
    CheckpointTransformBundle,
)
from worldscape_policy.types import WorldActionOutput


class _Transform:
    transforms = ()


class _Runtime:
    def __init__(self):
        self.policy = torch.nn.Linear(1, 1)
        self.pending = None
        self.commits = 0

    @property
    def has_pending_prediction(self):
        return self.pending is not None

    def reset(self, mode):
        self.mode = mode
        self.pending = None

    def predict(self, **kwargs):
        del kwargs
        self.pending = WorldActionOutput(action=torch.zeros(1, 4, 20))
        return self.pending

    def commit(self, output=None):
        assert output is self.pending
        self.pending = None
        self.commits += 1

    def discard(self):
        self.pending = None


class _FakeHDF5Robot:
    def __init__(self):
        self.cursor = 0
        self.sends = []

    def reset_episode(self):
        self.cursor = 0

    def read_observation(self, **kwargs):
        del kwargs
        self.cursor += 1
        image = np.full((1, 8, 8, 3), self.cursor, dtype=np.uint8)
        state = {
            "state.left_pos": np.zeros((1, 3), dtype=np.float32),
            "state.left_rot6d": np.array(
                [[1, 0, 0, 1, 0, 0]], dtype=np.float32
            ),
            "state.left_gripper": np.zeros((1, 1), dtype=np.float32),
            "state.right_pos": np.zeros((1, 3), dtype=np.float32),
            "state.right_rot6d": np.array(
                [[1, 0, 0, 1, 0, 0]], dtype=np.float32
            ),
            "state.right_gripper": np.zeros((1, 1), dtype=np.float32),
        }
        return image, image, image, state, self.cursor

    def send_end_pose_action(self, *args):
        self.sends.append(args)


class _ActionAdapter:
    def __call__(self, output):
        del output
        side = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * 4
        return side, [row.copy() for row in side]


def test_fake_hdf5_dry_run_produces_common_result_artifacts(tmp_path):
    runtime = _Runtime()
    robot = _FakeHDF5Robot()
    bundle = CheckpointTransformBundle(_Transform(), "agilex", 33, 64, 32)
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            bundle,
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
            demo_video=np.zeros((50, 8, 8, 3), dtype=np.uint8),
            session_id="fake-hdf5",
        ),
        _ActionAdapter(),
        artifact_writer=EvaluationArtifactWriter(tmp_path),
    )

    runner.run(
        RealRobotRunnerConfig(
            "interactive", max_steps=1, rollout_steps=2, dry_run=True
        ),
        generator=torch.Generator(),
    )

    assert runtime.commits == 1
    assert robot.sends == []
    record = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert record["status"] == "completed"
    assert record["completed_steps"] == 1
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "action_previews.jsonl").is_file()
