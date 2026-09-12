import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evals.agilex.action_adapter import AgileXActionAdapter
from evals.agilex.evaluate import (
    RealRobotRunner,
    RealRobotRunnerConfig,
    RobotResetRequiredError,
    TransportExecutionResult,
)
from evals.agilex.observation_adapter import (
    AgileXObservationAdapter,
    manifold_eef6d_state,
)
from evals.agilex.safety import SafetyGuard
from evals.common.artifacts import EvaluationArtifactWriter
from evals.common.checkpoint_runtime import (
    CheckpointTransformBundle,
    resolve_embodiment_id,
)
from worldscape_policy.types import (
    VisualMemoryState,
    WAMInferenceState,
    WorldActionOutput,
)


class FakeTransform:
    transforms = ()


def test_result_snapshot_drops_runtime_caches():
    output = WorldActionOutput(
        action=torch.ones(1, 2, 3),
        next_visual_memory=VisualMemoryState(
            wam_state=WAMInferenceState(
                positive_kv_cache=[torch.ones(2, 1, 4)]
            )
        ),
    )

    snapshot = output.result_snapshot()

    torch.testing.assert_close(snapshot.require_action(), output.require_action())
    assert snapshot.next_visual_memory is None
    assert output.next_visual_memory is not None


def test_manifold_eef_state_accepts_appended_gripper_pose_value():
    observation = {
        "left_end_pose": [[1, 2, 3, 0, 0, 0, 1, 0.08]],
        "right_end_pose": [[4, 5, 6, 0, 0, 0, 1, 0.04]],
        "left_arm_joint_state": [[0, 0, 0, 0.02]],
        "right_arm_joint_state": [[0, 0, 0, 0.03]],
    }

    state = manifold_eef6d_state(observation)

    np.testing.assert_allclose(state["state.left_pos"], [[1, 2, 3]])
    np.testing.assert_allclose(state["state.right_pos"], [[4, 5, 6]])
    np.testing.assert_allclose(state["state.left_gripper"], [[0.02]])
    np.testing.assert_allclose(state["state.right_gripper"], [[0.03]])


class FakeRuntime:
    def __init__(self, output):
        self.policy = torch.nn.Linear(1, 1)
        self.output = output
        self.pending = None
        self.commits = 0
        self.discards = 0
        self.predictions = []

    @property
    def has_pending_prediction(self):
        return self.pending is not None

    def reset(self, mode):
        self.mode = mode
        self.pending = None

    def predict(self, **kwargs):
        self.predictions.append(kwargs)
        self.pending = self.output
        return self.output

    def commit(self, output=None):
        assert output is self.pending
        self.commits += 1
        self.pending = None

    def discard(self):
        self.discards += 1
        self.pending = None


class FakeHDF5Robot:
    def __init__(self, *, send_error=None, fail_on_send=1):
        self.reads = 0
        self.sends = []
        self.send_error = send_error
        self.fail_on_send = fail_on_send
        self.send_attempts = 0
        self.resets = 0

    def reset_episode(self):
        self.resets += 1

    def read_observation(self, **kwargs):
        assert kwargs["action_mode"] == "eef"
        self.reads += 1
        image = np.full((1, 4, 5, 3), self.reads, dtype=np.uint8)
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
        return image, image, image, state, self.reads

    def send_end_pose_action(self, timestamp, rate, left, right):
        self.send_attempts += 1
        if (
            self.send_error is not None
            and self.send_attempts == self.fail_on_send
        ):
            raise self.send_error
        self.sends.append((timestamp, rate, left, right))


class TimingOutHDF5Robot(FakeHDF5Robot):
    def try_read_observation(self, **kwargs):
        if self.reads == 0:
            return super().read_observation(**kwargs)
        return None


class DelayedHDF5Robot(FakeHDF5Robot):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def try_read_observation(self, **kwargs):
        self.attempts += 1
        if self.attempts < 4:
            return None
        return super().read_observation(**kwargs)


class TimeoutAwareHDF5Robot(FakeHDF5Robot):
    def __init__(self, result, *, clock=None):
        super().__init__()
        self.result = result
        self.clock = clock
        self.timeout_calls = []

    def send_end_pose_action_with_timeout(
        self, timestamp, rate, left, right, *, timeout_s
    ):
        self.timeout_calls.append((timestamp, rate, left, right, timeout_s))
        if self.clock is not None:
            self.clock.value += timeout_s * 10
        if self.result.accepted:
            self.sends.append((timestamp, rate, left, right))
        return self.result


class BlockingLegacyHDF5Robot(FakeHDF5Robot):
    def send_end_pose_action(self, timestamp, rate, left, right):
        del timestamp, rate, left, right
        raise AssertionError("blocking legacy send must not be called")


class IncrementingClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.1
        return self.value


class FakeActionAdapter:
    def __init__(self, unsafe=False, horizon=4):
        self.unsafe = unsafe
        self.horizon = horizon

    def __call__(self, output):
        del output
        step = 1.0 if self.unsafe else 0.01
        left = [
            [index * step, 0, 0, 0, 0, 0, 1, 0.0]
            for index in range(self.horizon)
        ]
        right = [row.copy() for row in left]
        return left, right


class RelativeActionTransform:
    transforms = ()
    modality_metadata = {
        "action.left_pos": SimpleNamespace(absolute=False),
        "action.left_rot6d": SimpleNamespace(absolute=False),
        "action.right_pos": SimpleNamespace(absolute=False),
        "action.right_rot6d": SimpleNamespace(absolute=False),
    }

    def unapply(self, data):
        action = data["action"]
        horizon = action.shape[0]
        rot6d = torch.tensor([0, -1, 1, 0, 0, 0], dtype=torch.float32).repeat(
            horizon, 1
        )
        return {
            "action.left_pos": action[:, :3],
            "action.left_rot6d": rot6d,
            "action.left_gripper": torch.zeros(horizon, 1),
            "action.right_pos": action[:, 3:6],
            "action.right_rot6d": rot6d.clone(),
            "action.right_gripper": torch.zeros(horizon, 1),
        }


def _bundle():
    return CheckpointTransformBundle(
        transform=FakeTransform(),
        embodiment="agilex",
        embodiment_id=33,
        max_state_dim=64,
        max_action_dim=32,
    )


def test_observation_adapter_builds_explicit_range_and_embodiment():
    robot = FakeHDF5Robot()
    adapter = AgileXObservationAdapter(
        _bundle(),
        mode="interactive",
        instruction="fold",
        visual_input_range="minus_one_one",
        device="cpu",
    )

    result = adapter(robot.read_observation(
        use_history=False, num_history_frames=1, action_mode="eef"
    ))

    assert result.observation.images.shape == (1, 1, 3, 3, 4, 5)
    assert result.observation.proprioception.shape == (1, 1, 64)
    assert result.observation.embodiment_id.item() == 33
    assert result.prompts.language_instruction == ["fold"]
    assert result.observation.images.max().item() == pytest.approx(2 / 255 - 1)


def test_auto_observation_prompt_matches_training_without_negative_vlm_text():
    robot = FakeHDF5Robot()
    adapter = AgileXObservationAdapter(
        _bundle(),
        mode="auto",
        instruction="fold",
        visual_input_range="uint8",
        device="cpu",
    )

    result = adapter(robot.read_observation(
        use_history=False, num_history_frames=1, action_mode="eef"
    ))

    assert result.prompts.vlm_planning_text == ["fold"]
    assert result.prompts.negative_vlm_text is None


def test_observation_adapter_emits_session_visual_prompt_once_and_after_reset():
    robot = FakeHDF5Robot()
    goal = np.full((4, 5, 3), 255, dtype=np.uint8)
    adapter = AgileXObservationAdapter(
        _bundle(),
        mode="interactive",
        instruction="fold",
        visual_input_range="zero_one",
        device="cpu",
        goal_image=goal,
        session_id="demo-7",
    )
    raw = robot.read_observation(
        use_history=False, num_history_frames=1, action_mode="eef"
    )

    first = adapter(raw)
    second = adapter(raw)
    adapter.reset_episode()
    reset = adapter(raw)

    assert adapter.session_id == "demo-7"
    assert first.prompts.goal_images.shape == (1, 1, 3, 4, 5)
    assert first.prompts.goal_images.max().item() == 1.0
    assert second.prompts.goal_images is None
    assert reset.prompts.goal_images is not None


def test_action_adapter_unapplies_and_restores_relative_actions(monkeypatch):
    captured = {}

    def trajectories(action_fields, *, action_mode):
        assert action_mode == "eef"
        captured.update(action_fields)
        return (
            action_fields["action.left_pos"].tolist(),
            action_fields["action.right_pos"].tolist(),
        )

    monkeypatch.setattr(
        "evals.agilex.action_adapter.action_fields_to_trajectories",
        trajectories,
    )
    bundle = CheckpointTransformBundle(
        transform=RelativeActionTransform(),
        embodiment="agilex",
        embodiment_id=33,
        max_state_dim=64,
        max_action_dim=32,
    )
    left, right = AgileXActionAdapter(bundle)(
        WorldActionOutput(action=torch.zeros(1, 2, 6)),
        current_state={
            "state.left_pos": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            "state.left_rot6d": np.array(
                [[0, -1, 1, 0, 0, 0]], dtype=np.float32
            ),
            "state.right_pos": np.array([[0.4, 0.5, 0.6]], dtype=np.float32),
            "state.right_rot6d": np.array(
                [[0, -1, 1, 0, 0, 0]], dtype=np.float32
            ),
        },
    )

    np.testing.assert_allclose(left, [[0.1, 0.2, 0.3]] * 2)
    np.testing.assert_allclose(right, [[0.4, 0.5, 0.6]] * 2)
    expected_rotation = np.array([-1, 0, 0, -1, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(
        captured["action.left_rot6d"],
        np.broadcast_to(expected_rotation, (2, 6)),
        atol=1e-6,
    )


def test_segmented_hdf5_runner_commits_only_after_all_segments_succeed():
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    robot = FakeHDF5Robot()
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
    )

    outputs = runner.run(
        RealRobotRunnerConfig("interactive", max_steps=1, rollout_steps=2),
        generator=torch.Generator(),
    )

    assert len(outputs) == 1
    assert outputs[0] is not output
    torch.testing.assert_close(outputs[0].require_action(), output.require_action())
    assert outputs[0].next_visual_memory is None
    assert len(robot.sends) == 2
    assert robot.reads == 3
    assert runtime.commits == 1
    assert runtime.discards == 0
    assert robot.resets == 1


def test_segmented_rollout_prefills_next_plan_with_gt_observation_chunk():
    output = WorldActionOutput(action=torch.zeros(1, 24, 20))
    runtime = FakeRuntime(output)
    runner = RealRobotRunner(
        runtime,
        FakeHDF5Robot(),
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(horizon=24),
    )

    runner.run(
        RealRobotRunnerConfig(
            "interactive",
            max_steps=2,
            rollout_steps=8,
            use_history=False,
        ),
        generator=torch.Generator(),
    )

    assert len(runtime.predictions) == 2
    assert runtime.predictions[0]["observation"].images.shape[1] == 1
    # One initial frame plus one real observation after each execution segment.
    assert runtime.predictions[1]["observation"].images.shape[1] == 9


def test_dry_run_previews_without_sending_and_writes_common_artifacts(tmp_path):
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    robot = FakeHDF5Robot()
    previews = []
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
        artifact_writer=EvaluationArtifactWriter(tmp_path),
        action_preview=previews.append,
    )

    runner.run(
        RealRobotRunnerConfig(
            "interactive", max_steps=1, rollout_steps=2, dry_run=True
        ),
        generator=torch.Generator(),
    )

    assert robot.sends == []
    assert runtime.commits == 1
    assert len(previews) == 1
    assert previews[0].dry_run
    episode = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert episode["status"] == "completed"
    assert episode["latency"]["prediction"]["count"] == 1
    preview = json.loads((tmp_path / "action_previews.jsonl").read_text())
    assert preview["record_type"] == "agilex_action_preview"


def test_frame_capture_is_bounded_and_passed_to_video_writer(
    tmp_path, monkeypatch
):
    import imageio.v3 as iio

    written = {}

    def fake_imwrite(path, frames, *, fps):
        written.update(path=path, frames=frames, fps=fps)

    monkeypatch.setattr(iio, "imwrite", fake_imwrite)
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runner = RealRobotRunner(
        FakeRuntime(output),
        FakeHDF5Robot(),
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
        artifact_writer=EvaluationArtifactWriter(tmp_path, video_fps=12),
    )

    runner.run(
        RealRobotRunnerConfig(
            "interactive",
            max_steps=1,
            rollout_steps=2,
            dry_run=True,
            capture_frames=True,
            max_recorded_frames=2,
        ),
        generator=torch.Generator(),
    )

    assert runner.last_result is not None
    assert len(runner.last_result.episodes[0].frames) == 2
    assert written["path"] == tmp_path / "videos" / "agilex-0000.mp4"
    assert written["fps"] == 12
    assert written["frames"].shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(written["frames"][:, 0, 0, 0], [1, 2])


def test_frame_capture_is_disabled_by_default():
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runner = RealRobotRunner(
        FakeRuntime(output),
        FakeHDF5Robot(),
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
    )

    runner.run(
        RealRobotRunnerConfig(
            "interactive", max_steps=2, rollout_steps=2, dry_run=True
        ),
        generator=torch.Generator(),
    )

    assert runner.last_result is not None
    assert runner.last_result.episodes[0].frames == ()


def test_failure_artifact_keeps_only_successfully_sampled_frames(
    tmp_path, monkeypatch
):
    import imageio.v3 as iio

    videos = []
    monkeypatch.setattr(
        iio,
        "imwrite",
        lambda path, frames, *, fps: videos.append((path, frames.copy(), fps)),
    )
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    runner = RealRobotRunner(
        runtime,
        FakeHDF5Robot(
            send_error=RuntimeError("second segment rejected"),
            fail_on_send=2,
        ),
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
        artifact_writer=EvaluationArtifactWriter(tmp_path),
    )

    with pytest.raises(RobotResetRequiredError):
        runner.run(
            RealRobotRunnerConfig(
                "interactive",
                max_steps=1,
                rollout_steps=2,
                capture_frames=True,
                max_recorded_frames=8,
            ),
            generator=torch.Generator(),
        )

    assert runtime.commits == 0
    assert runtime.discards == 1
    assert runner.last_result is not None
    episode = runner.last_result.episodes[0]
    assert episode.record.status == "failed"
    assert len(episode.frames) == 2
    assert len(videos) == 1
    np.testing.assert_array_equal(videos[0][1][:, 0, 0, 0], [1, 2])
    record = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert record["completed_steps"] == 0
    assert record["error_type"] == "RobotResetRequiredError"


def test_failure_is_logged_to_common_artifact_before_reraise(tmp_path, caplog):
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    runner = RealRobotRunner(
        runtime,
        FakeHDF5Robot(send_error=RuntimeError("transport rejected")),
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
        artifact_writer=EvaluationArtifactWriter(tmp_path),
    )

    with pytest.raises(RuntimeError, match="transport rejected"):
        runner.run(
            RealRobotRunnerConfig("interactive", max_steps=1, rollout_steps=2),
            generator=torch.Generator(),
        )

    record = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert record["steps"][0]["error_message"] == "transport rejected"
    assert "AgileX rollout failed" in caplog.text


def test_native_runner_discards_on_send_failure():
    failure = RuntimeError("send failed")
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    robot = FakeHDF5Robot(send_error=failure)
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
        safety_guard=SafetyGuard(max_position_step_m=0.2),
    )

    with pytest.raises(RuntimeError):
        runner.run(
            RealRobotRunnerConfig("interactive", max_steps=1, rollout_steps=2),
            generator=torch.Generator(),
        )

    assert runtime.commits == 0
    assert runtime.discards == 1
    assert not runtime.has_pending_prediction
    assert robot.sends == []


def test_native_runner_warns_and_sends_motion_limit_violation(caplog):
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    robot = FakeHDF5Robot()
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(unsafe=True),
        safety_guard=SafetyGuard(max_position_step_m=0.2),
    )

    outputs = runner.run(
        RealRobotRunnerConfig("interactive", max_steps=1, rollout_steps=2),
        generator=torch.Generator(),
    )

    assert len(outputs) == 1
    assert outputs[0] is not output
    torch.testing.assert_close(outputs[0].require_action(), output.require_action())
    assert outputs[0].next_visual_memory is None
    assert runtime.commits == 1
    assert runtime.discards == 0
    assert not runtime.has_pending_prediction
    assert len(robot.sends) == 2
    assert "command will still be sent" in caplog.text
    assert runner.last_result is not None
    assert runner.last_result.episodes[0].record.steps[0].status == "completed"


def test_native_runner_discards_on_segment_observation_timeout():
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    robot = TimingOutHDF5Robot()
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
        clock=IncrementingClock(),
        sleep=lambda _: None,
    )

    with pytest.raises(RobotResetRequiredError) as raised:
        runner.run(
            RealRobotRunnerConfig(
                "interactive",
                max_steps=1,
                rollout_steps=2,
                observation_timeout_s=0.15,
            ),
            generator=torch.Generator(),
        )

    assert isinstance(raised.value.__cause__, TimeoutError)
    assert raised.value.accepted_segments == 1
    assert len(robot.sends) == 1
    assert runtime.commits == 0
    assert runtime.discards == 1
    assert not runtime.has_pending_prediction


def test_native_runner_can_retry_after_observation_timeout():
    robot = DelayedHDF5Robot()
    runner = RealRobotRunner(
        FakeRuntime(WorldActionOutput(action=torch.zeros(1, 4, 20))),
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
        clock=IncrementingClock(),
        sleep=lambda _: None,
    )

    result = runner._read(
        RealRobotRunnerConfig(
            "interactive",
            observation_timeout_s=0.15,
            observation_timeout_policy="retry",
        ),
        step=0,
    )

    assert result[-1] == 1
    assert robot.attempts == 4


def test_native_runner_discards_on_send_failure_after_partial_prefix():
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    robot = FakeHDF5Robot(
        send_error=RuntimeError("second segment rejected"),
        fail_on_send=2,
    )
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
    )

    with pytest.raises(RobotResetRequiredError) as raised:
        runner.run(
            RealRobotRunnerConfig("interactive", max_steps=1, rollout_steps=2),
            generator=torch.Generator(),
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert raised.value.accepted_segments == 1
    assert len(robot.sends) == 1
    assert runtime.commits == 0
    assert runtime.discards == 1
    assert not runtime.has_pending_prediction


def test_send_timeout_fails_closed_before_blocking_legacy_transport():
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    robot = BlockingLegacyHDF5Robot()
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
    )

    with pytest.raises(RuntimeError, match="requires the robot transport"):
        runner.run(
            RealRobotRunnerConfig(
                "interactive", max_steps=1, send_timeout_s=0.01
            ),
            generator=torch.Generator(),
        )

    assert robot.sends == []
    assert runtime.commits == 0
    assert runtime.discards == 1
    assert not runtime.has_pending_prediction


def test_timeout_aware_transport_timeout_is_unexecuted():
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    robot = TimeoutAwareHDF5Robot(
        TransportExecutionResult(
            accepted=False, timed_out=True, detail="deadline expired"
        )
    )
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
    )

    with pytest.raises(TimeoutError, match="deadline expired"):
        runner.run(
            RealRobotRunnerConfig(
                "interactive", max_steps=1, send_timeout_s=0.01
            ),
            generator=torch.Generator(),
        )

    assert robot.timeout_calls[0][-1] == pytest.approx(0.01)
    assert runtime.commits == 0
    assert runtime.discards == 1
    assert not runtime.has_pending_prediction


def test_acknowledged_slow_send_commits_instead_of_timing_out():
    output = WorldActionOutput(action=torch.zeros(1, 4, 20))
    runtime = FakeRuntime(output)
    clock = IncrementingClock()
    robot = TimeoutAwareHDF5Robot(
        TransportExecutionResult(accepted=True),
        clock=clock,
    )
    runner = RealRobotRunner(
        runtime,
        robot,
        AgileXObservationAdapter(
            _bundle(),
            mode="interactive",
            instruction="fold",
            visual_input_range="uint8",
            device="cpu",
        ),
        FakeActionAdapter(),
        clock=clock,
    )

    outputs = runner.run(
        RealRobotRunnerConfig(
            "interactive",
            max_steps=1,
            rollout_steps=1,
            send_timeout_s=0.01,
        ),
        generator=torch.Generator(),
    )

    assert len(outputs) == 1
    assert outputs[0] is not output
    torch.testing.assert_close(outputs[0].require_action(), output.require_action())
    assert outputs[0].next_visual_memory is None
    assert len(robot.sends) == 1
    assert runtime.commits == 1
    assert runtime.discards == 0


def test_embodiment_resolution_is_checkpoint_owned_and_fail_closed():
    assert resolve_embodiment_id({"agilex": 33}, "agilex") == 33
    assert resolve_embodiment_id({"lerobot_eef_lctx": 33}, "agilex") == 33
    assert resolve_embodiment_id({"lerobot_eef": 33}, "lerobot_eef") == 33
    with pytest.raises(KeyError):
        resolve_embodiment_id({}, "agilex")
