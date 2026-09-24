import json

import numpy as np
import pytest
import torch

import evals.common.hdf5_replay as hdf5_replay
from evals.agilex.robot import (
    AgileXRobotObservation,
)
from evals.common.hdf5_replay import (
    HDF5ReplayConfig,
    run_hdf5_replay,
)
from evals.common.protocols import RolloutInput
from worldscape_policy.rollout.runner import (
    RolloutConfig,
    RolloutRunner,
)
from worldscape_policy.types import (
    ObservationBatch,
    PromptBatch,
    WorldActionOutput,
)


class FakeRuntime:
    def __init__(self, outputs):
        self.policy = torch.nn.Linear(1, 1)
        self.outputs = iter(outputs)
        self.pending = None
        self.committed = []
        self.discard_count = 0
        self.reset_modes = []

    @property
    def has_pending_prediction(self):
        return self.pending is not None

    def reset(self, mode):
        self.reset_modes.append(mode)
        self.pending = None

    def predict(self, *, observation, prompts, generator):
        del observation, prompts, generator
        self.pending = next(self.outputs)
        return self.pending

    def commit(self, output=None):
        assert output is self.pending
        self.committed.append(self.pending)
        self.pending = None

    def discard(self):
        self.discard_count += 1
        self.pending = None


class FakeObservationSource:
    def __init__(self):
        self.steps = []

    def read(self, step_index):
        self.steps.append(step_index)
        return RolloutInput(
            observation=ObservationBatch(
                images=torch.zeros(1, 1, 1, 3, 2, 2),
                head_view=torch.zeros(1, 1, 3, 2, 2),
                proprioception=torch.zeros(1, 1, 2),
                embodiment_id=torch.zeros(1, dtype=torch.long),
            ),
            prompts=PromptBatch(vlm_planning_text=["test"]),
        )


class FakeExecutor:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def execute(self, output, *, timeout_s):
        self.calls.append((output, timeout_s))
        if self.error is not None:
            raise self.error


class IncrementingClock:
    def __init__(self, increment=0.001):
        self.value = 0.0
        self.increment = increment

    def __call__(self):
        value = self.value
        self.value += self.increment
        return value


def test_runner_commits_only_after_successful_execution():
    outputs = [WorldActionOutput(action=torch.ones(1)) for _ in range(2)]
    runtime = FakeRuntime(outputs)
    source = FakeObservationSource()
    executor = FakeExecutor()

    result = RolloutRunner(
        runtime,
        source,
        executor,
        clock=IncrementingClock(),
    ).run(
        RolloutConfig("auto", max_steps=2, episode_id="episode-1"),
        generator=torch.Generator(),
    )

    assert list(result.outputs) == outputs
    assert runtime.committed == outputs
    assert runtime.discard_count == 0
    assert source.steps == [0, 1]
    assert result.record.status == "completed"
    assert result.record.completed_steps == 2
    assert result.record.latency.prediction.count == 2
    assert result.record.latency.prediction.mean_ms == pytest.approx(1.0)
    assert json.loads(result.record.to_json())["episode_id"] == "episode-1"
    assert "steps" not in result.record.summary()
    assert [record["record_type"] for record in result.jsonl_records()] == [
        "rollout_step",
        "rollout_step",
        "rollout_episode",
    ]


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (RuntimeError("execution failed"), "failed"),
        (TimeoutError("execution deadline exceeded"), "timed_out"),
    ],
)
def test_runner_discards_pending_prediction_on_execution_failure(
    error,
    expected_status,
):
    output = WorldActionOutput(action=torch.ones(1))
    runtime = FakeRuntime([output])
    executor = FakeExecutor(error)

    result = RolloutRunner(
        runtime,
        FakeObservationSource(),
        executor,
        clock=IncrementingClock(),
    ).run(
        RolloutConfig(
            "interactive",
            max_steps=3,
            episode_id="failed-episode",
            execution_timeout_s=0.5,
        ),
        generator=torch.Generator(),
    )

    assert result.outputs == ()
    assert runtime.committed == []
    assert runtime.discard_count == 1
    assert not runtime.has_pending_prediction
    assert result.error is error
    assert result.record.status == expected_status
    assert result.record.steps[0].status == expected_status
    assert executor.calls == [(output, 0.5)]
    with pytest.raises(type(error), match=str(error)):
        result.raise_for_error()


def test_hdf5_replay_keeps_list_api_over_common_runner(monkeypatch):
    video = np.zeros((1, 2, 2, 3), dtype=np.uint8)

    class FakeRobot:
        def __init__(self, path):
            assert path == "episode.hdf5"

        def default_max_steps(self, rollout_steps):
            assert rollout_steps == 1
            return 2

        def observe(self, request):
            assert request.use_history is True
            assert request.num_history_frames == 4
            assert request.action_mode == "eef"
            return AgileXRobotObservation(
                high=video,
                left=video,
                right=video,
                state={"state.left_joint": np.zeros((1, 7), dtype=np.float32)},
                timestamp=None,
            )

    monkeypatch.setattr(hdf5_replay, "HDF5ReplayRobot", FakeRobot)
    outputs = [WorldActionOutput(action=torch.ones(1)) for _ in range(2)]
    runtime = FakeRuntime(outputs)

    result = run_hdf5_replay(
        runtime,
        HDF5ReplayConfig(
            "episode.hdf5",
            "auto",
            "pick",
            1,
            max_steps=2,
        ),
        generator=torch.Generator(),
    )

    assert result == outputs
    assert runtime.committed == outputs
