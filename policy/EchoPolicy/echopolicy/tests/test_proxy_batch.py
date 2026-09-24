import asyncio
import json
import pytest

import numpy as np
from PIL import Image

from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig, _joint_state_from_action
from vlm_orchestrator.strategies.base import SessionState
from vlm_orchestrator.strategies.passthrough import PassthroughStrategy
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy
from vlm_orchestrator.strategies.base import StrategyContext
from vlm_orchestrator.vlm import PassthroughVLM
from vlm_orchestrator.vlm.api import DecompositionResult, GoogleVLM
from vlm_orchestrator.trajectory_selection import group_trajectory_endpoints


def test_endpoint_groups_separate_either_arm_and_do_not_chain():
    projected = [{"left": [[x, 0]], "right": [[right, 0]]}
                 for x, right in [(0, 0), (10, 100), (20, 0)]]
    assert [group["candidate_indices"] for group in group_trajectory_endpoints(projected, "left")] == [[0, 1], [2]]
    assert [group["candidate_indices"] for group in group_trajectory_endpoints(projected)] == [[0], [1], [2]]
    same_active_arm = [{"left": [[0, 0]], "right": [[x, 0]]} for x in (0, 10, 20)]
    assert [group["candidate_indices"] for group in group_trajectory_endpoints(same_active_arm, radius=15)] == [[0, 1], [2]]
    with pytest.raises(ValueError):
        group_trajectory_endpoints([{"left": [[float("nan"), 0]]}], "left")


def test_vla_candidates_use_last_dispatched_joint_states():
    state = SessionState(last_dispatched_joint_states=[
        (np.arange(14, dtype=float) - 100).tolist(),
        (np.arange(14, dtype=float) - 50).tolist(),
        np.arange(14, dtype=float).tolist(),
        (np.arange(14, dtype=float) + 100).tolist(),
        (np.arange(14, dtype=float) + 200).tolist(),
    ])
    obs = {
        "observation/joint_position": np.zeros(12, dtype=np.float32),
        "observation/gripper_position": np.zeros(2, dtype=np.float32),
    }
    candidates = OrchestratorProxy._candidate_vla_observations(obs, state, 15)
    np.testing.assert_array_equal(
        candidates[0]["observation/joint_position"],
        np.asarray([-100, -99, -98, -97, -96, -95, -93, -92, -91, -90, -89, -88], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        candidates[0]["observation/gripper_position"],
        np.asarray([-94, -87], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        candidates[1]["observation/joint_position"],
        candidates[0]["observation/joint_position"],
    )
    np.testing.assert_array_equal(candidates[3]["observation/joint_position"], np.asarray([-50, -49, -48, -47, -46, -45, -43, -42, -41, -40, -39, -38], dtype=np.float32))
    np.testing.assert_array_equal(candidates[6]["observation/joint_position"], np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.float32))
    np.testing.assert_array_equal(candidates[9]["observation/joint_position"], np.asarray([100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112], dtype=np.float32))
    np.testing.assert_array_equal(candidates[12]["observation/joint_position"], np.asarray([200, 201, 202, 203, 204, 205, 207, 208, 209, 210, 211, 212], dtype=np.float32))


def test_vla_candidates_add_independent_noise_to_rgb_images_only():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    depth = np.full((16, 16), 0.5, dtype=np.float32)
    obs = {
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": image.copy(),
        "observation/depth_exterior_image_1_left": depth,
    }

    candidates = OrchestratorProxy._candidate_vla_observations(
        obs, SessionState(), 3, image_noise_std=2.0,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_array_equal(obs["observation/exterior_image_1_left"], image)
    np.testing.assert_array_equal(candidates[0]["observation/depth_exterior_image_1_left"], depth)
    assert all(candidate["observation/exterior_image_1_left"].dtype == np.uint8 for candidate in candidates)
    assert all(not np.array_equal(candidate["observation/exterior_image_1_left"], image) for candidate in candidates)
    assert not np.array_equal(
        candidates[0]["observation/exterior_image_1_left"],
        candidates[1]["observation/exterior_image_1_left"],
    )
    assert not np.array_equal(
        candidates[0]["observation/exterior_image_1_left"],
        candidates[0]["observation/wrist_image_left"],
    )


def test_vla_candidate_image_noise_can_be_disabled():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    obs = {"observation/exterior_image_1_left": image}

    candidate = OrchestratorProxy._candidate_vla_observations(
        obs, SessionState(), 1, image_noise_std=0.0,
    )[0]

    np.testing.assert_array_equal(candidate["observation/exterior_image_1_left"], image)


def test_vla_candidates_add_joint_noise_only_for_far_targets():
    state = SessionState(last_dispatched_joint_states=[np.zeros(14).tolist()] * 5)
    obs = {
        "observation/joint_position": np.zeros(12, dtype=np.float32),
        "observation/gripper_position": np.zeros(2, dtype=np.float32),
    }
    near = OrchestratorProxy._candidate_vla_observations(
        obs, state, 3, joint_state_noise_std=0.1, far_target=False,
        rng=np.random.default_rng(4),
    )
    far = OrchestratorProxy._candidate_vla_observations(
        obs, state, 3, joint_state_noise_std=0.1, far_target=True,
        rng=np.random.default_rng(4),
    )
    assert all(np.array_equal(item["observation/joint_position"], near[0]["observation/joint_position"]) for item in near)
    assert any(not np.array_equal(item["observation/joint_position"], near[index]["observation/joint_position"]) for index, item in enumerate(far))
    np.testing.assert_array_equal(far[0]["observation/gripper_position"], np.zeros(2, dtype=np.float32))


def test_joint_state_from_dispatched_array_uses_robo_dojo_layout():
    action = np.arange(14, dtype=float)
    assert _joint_state_from_action(action) == action.tolist()


def test_vlm_group_choice_controls_output_and_fixed_10_step_window(monkeypatch, tmp_path):
    steps = 10
    class ChoosingVLM(PassthroughVLM):
        def select_trajectory_group(self, instruction, subgoal, groups, image, extra_images=None, *, before_images=None, previous_available=False, memory="", last_executed_ee_trajectory=None, **kwargs):
            assert instruction == "high level task"
            hidden = {
                "candidate_indices", "candidate_sources", "endpoints",
                "trajectory_z_ranges_m", "representative_candidate_indices",
                "representative_support_count", "representative_average_action_distance",
            }
            assert all(not (hidden & set(group)) for group in groups)
            assert all("member_candidate_indices" not in group for group in groups)
            assert all("mean_ee_endpoints" not in group for group in groups)
            assert all("mean_ee_trajectory_z_ranges_m" not in group for group in groups)
            assert all("mean_ee_trajectory" in group for group in groups)
            assert all("mean_ee_trajectory_steps" in group for group in groups)
            assert all(
                set(group["mean_ee_trajectory"]) == {"left", "right"}
                for group in groups
            )
            assert all(
                len(group["mean_ee_trajectory"][arm]) == group["mean_ee_trajectory_steps"][arm]
                for group in groups for arm in ("left", "right")
            )
            assert all(group["overlay_trajectory_steps"] == 30 for group in groups)
            assert all(group["trajectory_steps"] == steps for group in groups)
            assert all(group["atomic_action"] == "pick" for group in groups)
            assert all(set(group["current_ee"]) == {"left", "right"} for group in groups)
            assert all(
                len(group["current_ee"][arm]["position_world_m"]) == 3
                for group in groups for arm in ("left", "right")
            )
            assert all(
                len(group["current_ee"][arm]["rotation_world_matrix"]) == 3
                for group in groups for arm in ("left", "right")
            )
            assert memory == "The target is securely grasped and still needs lifting."
            assert last_executed_ee_trajectory == {
                "executed_steps": 10,
                "steps": [],
            }
            assert len(extra_images) == 5  # two wrists, two groups, and previous
            assert all(item.shape == image.shape for item in extra_images[-3:])
            assert len(before_images) == 3
            return {"group_id": 1, "reason": "correct object"}
    strategy = PassthroughStrategy(StrategyContext(vlm=ChoosingVLM()))
    proxy = OrchestratorProxy(ProxyConfig(strategy=strategy))
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate",
                        lambda chunks, **kwargs: (0, {"trajectory_steps_requested": steps}))
    def project(chunks, *args):
        return [{"left": [[0, 0], [100 if np.asarray(chunk)[0, 0] > 0.5 else 10, 20]],
                 "right": [[0, 0], [0, 10]]} for chunk in chunks]
    monkeypatch.setattr(proxy, "_project_candidate_trajectories", project)
    state = SessionState(original_instruction="high level task", subgoals=["pick target"],
                         subgoal_arms=["left"], subgoal_atomic_actions=["pick"],
                         subgoal_coordinates=[[[100, 20]]], episode_log_dir=str(tmp_path),
                         action_cache=np.zeros((50, 14)), action_cache_target_revision=-1)
    state.check_memory = "The target is securely grasped and still needs lifting."
    state.last_executed_ee_trajectory = {
        "executed_steps": 10,
        "steps": [],
    }
    obs = _observation(0)
    obs["observation/joint_position"] = np.zeros(12)
    response = proxy._select_grouped_vla_response(
        [{"actions": np.full((50, 14), value)} for value in [0, 0.01, 1]], obs, state,
        [obs[proxy.config.image_key]] * 3,
    )
    np.testing.assert_array_equal(response["actions"], np.ones((50, 14)))
    selection = response["video_overlay"]["trajectory_selection"]
    assert selection["selected_group_id"] == 1
    assert selection["selected_candidate_index"] is None
    assert selection["representative_candidate_index"] == 2
    assert selection["consistent_candidate_indices"] == [2]
    assert selection["selected_action_steps"] == 50
    dispatched = proxy._cached_action_chunk(response, state, 10)
    assert len(dispatched["actions"]) == 10
    assert len(proxy._previous_candidate_remainder(state)) == 40
    assert (tmp_path / "vla_overlay_000000.png").exists()
    selection_image = tmp_path / "vla_group_selection_000000.png"
    assert selection_image.exists()
    assert selection["group_selection_image_path"] == str(selection_image)
    with Image.open(selection_image) as saved:
        assert saved.width == 24  # two new groups plus the PREVIOUS panel
        assert saved.height > 8  # selection reason header plus group panels


def test_previous_trajectory_remainder_is_grouped_as_candidate(monkeypatch):
    class ChoosingVLM(PassthroughVLM):
        def select_trajectory_group(self, instruction, subgoal, groups, image, extra_images=None, *, before_images=None, previous_available=False, memory="", last_executed_ee_trajectory=None, **kwargs):
            assert len(groups) == 2
            previous_group = next(
                group for group in groups if group["contains_previous_remaining"]
            )
            assert sum(group["contains_previous_remaining"] for group in groups) == 1
            assert "candidate_indices" not in previous_group
            assert "candidate_sources" not in previous_group
            assert previous_available is False
            assert len(extra_images) == 4  # two wrists, then G0 and G1
            return {"group_id": previous_group["group_id"], "reason": "continue previous"}
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=PassthroughStrategy(StrategyContext(vlm=ChoosingVLM())),
    ))
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate",
                        lambda chunks, **kwargs: (0, {"trajectory_steps_requested": 30}))
    def same_endpoint_fk(chunk):
        count = len(chunk)
        trajectory = np.zeros((count, 3), dtype=float)
        return {"left": trajectory.copy(), "right": trajectory.copy()}
    monkeypatch.setattr("vlm_orchestrator.proxy.fk_trajectory", same_endpoint_fk)
    monkeypatch.setattr(proxy, "_project_candidate_trajectories",
                        lambda chunks, *args: [{"left": [[0, 0], [100 if np.asarray(chunk)[0, 0] > 0 else 10, 0]],
                                                "right": [[0, 0], [0, 0]]} for chunk in chunks])
    old = np.repeat(np.arange(30, dtype=float)[:, None], 14, axis=1)
    state = SessionState(subgoals=["pick middle"], subgoal_arms=["left"],
                         subgoal_atomic_actions=["pick"], subgoal_coordinates=[[[100, 0]]],
                         action_cache=old.copy(), action_cache_offset=10,
                         action_cache_target_revision=0, target_revision=0)
    candidates = [{"actions": np.zeros((50, 14))} for _ in range(15)]
    obs = _observation(0)
    obs["observation/exterior_image_1_left"] = np.full((64, 64, 3), 123, dtype=np.uint8)
    response = proxy._select_grouped_vla_response(candidates, obs, state)
    np.testing.assert_array_equal(response["actions"], old[10:])
    selection = response["video_overlay"]["trajectory_selection"]
    assert selection["previous_option_selected"] is False
    assert selection["previous_steps_available"] == 20
    assert selection["previous_cache_offset"] == 10
    assert selection["previous_candidate_index"] == 15
    assert selection["previous_candidate_group_id"] == selection["selected_group_id"]
    assert selection["previous_candidate_selected"] is True
    previous_group = selection["groups"][selection["previous_candidate_group_id"]]
    assert previous_group["candidate_indices"] == [15]
    assert previous_group["trajectory_steps"] == 10
    assert all(
        15 not in group["candidate_indices"]
        for group in selection["groups"]
        if group["group_id"] != selection["previous_candidate_group_id"]
    )
    assert selection["candidate_sources"] == ["new"] * 15 + ["previous_remaining"]
    assert selection["new_candidate_count"] == 15
    assert selection["grouped_candidate_count"] == 16
    assert len(candidates) == 15
    np.testing.assert_array_equal(state.action_cache, old)


@pytest.mark.parametrize("revision,offset,flush,has_previous", [
    (1, 10, False, True), (0, 30, False, False), (0, 10, True, False),
])
def test_exhausted_or_flushed_previous_trajectory_is_not_grouped(
    monkeypatch, revision, offset, flush, has_previous,
):
    class ChoosingVLM(PassthroughVLM):
        def select_trajectory_group(self, instruction, subgoal, groups, image, extra_images=None, *, before_images=None, previous_available=False, memory="", last_executed_ee_trajectory=None, **kwargs):
            assert len(groups) == (2 if has_previous else 1)
            previous = groups[-1]
            assert "candidate_indices" not in previous
            assert previous["contains_previous_remaining"] is has_previous
            assert previous_available is False
            return {"group_id": 0, "reason": "new trajectory"}
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=PassthroughStrategy(StrategyContext(vlm=ChoosingVLM())),
    ))
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate",
                        lambda chunks, **kwargs: (0, {"trajectory_steps_requested": 30}))
    monkeypatch.setattr(proxy, "_project_candidate_trajectories",
                        lambda chunks, *args: [{"left": [[0, 0], [10, 10]], "right": [[0, 0], [20, 20]]}
                                               for _ in chunks])
    old = np.repeat(np.arange(30, dtype=float)[:, None], 14, axis=1)
    state = SessionState(
        subgoals=["pick"], subgoal_arms=["left"],
        subgoal_atomic_actions=["pick"], subgoal_coordinates=[[[10, 10]]],
        action_cache=old, action_cache_offset=offset,
        action_cache_target_revision=0, target_revision=revision,
        flush_actions=flush,
    )
    response = proxy._select_grouped_vla_response(
        [{"actions": np.zeros((50, 14))}], _observation(0), state,
    )
    np.testing.assert_array_equal(response["actions"], np.zeros((50, 14)))
    selection = response["video_overlay"]["trajectory_selection"]
    assert (selection["previous_candidate_index"] is not None) is has_previous
    assert (selection["previous_steps_available"] > 0) is has_previous


def test_google_group_selection_image_order_and_invalid_id(monkeypatch):
    vlm = GoogleVLM(api_key="test")
    images = [np.full((8, 8, 3), value, dtype=np.uint8) for value in range(9)]
    groups = [
        {
            "group_id": 0,
            "atomic_action": "pick",
            "mean_ee_trajectory": {
                "left": [[0.0, 0.0, 0.9], [0.0, 0.0, 0.8]],
                "right": [[0.0, 0.0, 0.9], [0.0, 0.0, 0.8]],
            },
            "mean_ee_trajectory_steps": {"left": 2, "right": 2},
            "trajectory_steps": 10,
            "overlay_trajectory_steps": 30,
        },
        {
            "group_id": 1,
            "atomic_action": "pick",
            "mean_ee_trajectory": {
                "left": [[0.0, 0.0, 0.9], [0.1, 0.0, 0.8]],
                "right": [[0.0, 0.0, 0.9], [0.1, 0.0, 0.8]],
            },
            "mean_ee_trajectory_steps": {"left": 2, "right": 2},
            "trajectory_steps": 10,
            "overlay_trajectory_steps": 30,
        },
    ]
    def generate(system, text, image, extra_images, response_schema):
        assert [int(value[0, 0, 0]) for value in [image, *extra_images]] == list(range(9))
        assert "then 4 NOW views" in text
        assert "PREVIOUS selected trajectory" not in text
        assert "Current atomic action: pick" in text
        assert (
            "Current atomic action description: Grasp the specified object "
            "securely and lift it clear of the ground."
        ) in text
        assert "Current left/right EE poses:" in text
        assert "non-binding planning hint" in text
        assert "Accumulated check memory: grasp is secure; lift remains" in text
        assert "Official task context:" in text
        assert "Move the task cube to its target" in text
        assert "robot returns to origin" in text
        assert "Recent executed EE trajectory (up to 20 steps; each action chunk is normally 10 steps;" in text
        assert "previous_executed_chunk=preceding inference" in text
        assert "'executed_steps': 10" in text
        assert "Judge every displayed candidate group" in text
        assert "standalone unexecuted remainder" in text
        assert "mean_ee_trajectory_z_ranges_m" not in text
        assert "mean_ee_endpoints" not in text
        assert "'mean_ee_trajectory':" in text
        assert "'mean_ee_trajectory_steps':" in text
        assert (
            "Each group is represented by one precomputed mean action. The numeric group data contains the "
        ) in text
        assert "predicted future Z minimum" not in text
        assert "Z min/max" not in text
        assert (
            "The executable prefix may show only an initial approach and need not reach the "
            "target within the displayed horizon. Prefer a trajectory that makes clear "
            "directional progress toward the current target over a hold or retreat trajectory."
        ) in text
        assert "Completion decisions (decision, decision_reason, progress_reason, memory, and before_to_now_summary)" in text
        assert "Candidate group trajectories are future predictions and MUST NOT be used as evidence" in text
        assert (
            "After a subgoal changes, judge trajectories only against the new current subgoal. "
            "A trajectory that only retracts from the previous target is insufficient if another "
            "candidate makes clearer progress toward the current target. Do not justify a "
            "previous-target clearing motion as progress toward the new target."
        ) in text
        assert "Make both reason fields numeric whenever the supplied evidence contains numbers" in text
        assert "trajectory-selection reason must be concise but quantitative" in text
        assert "The corresponding group image is an overlay of that same mean action over 30 steps" in text
        assert "representative_support_count" not in text
        assert "representative_candidate_indices" not in text
        return (
            '{"group_id":0,"reason":"continue","decision":"stay",'
            '"progress_reason":"incomplete","memory":"unchanged",'
            '"before_to_now_summary":"no verified change",'
            '"replan":false}'
        )
    monkeypatch.setattr(vlm, "generate", generate)
    assert vlm.select_trajectory_group(
        "task", "pick", groups, images[3], images[4:],
        before_images=images[:3], previous_available=True,
        memory="grasp is secure; lift remains",
        last_executed_ee_trajectory={"executed_steps": 10},
        task_description="Move the task cube to its target.",
        full_score_condition="The cube is placed and the robot returns to origin.",
    )["group_id"] == 0
    monkeypatch.setattr(vlm, "generate", lambda *args: '{"group_id": 9}')
    with pytest.raises(ValueError):
        vlm.select_trajectory_group("task", "pick", groups, images[3], images[4:])


@pytest.mark.skip(reason="legacy IK/NEXT_STATE protocol removed")
def test_google_group_selection_requires_next_state(monkeypatch):
    vlm = GoogleVLM(api_key="test")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    def generate(system, text, *args):
        assert "Current atomic action: insert" in text
        assert "fails to make progress on the atomic action" in text
        next_state_schema = args[-1]["properties"]["next_state"]
        properties = next_state_schema["properties"]
        assert properties["orientation_reference"]["enum"] == ["image"]
        assert "orientation_mode" not in properties
        assert "position_reference" not in properties
        assert "offset_world_m" not in properties
        assert {"target_pixel", "orientation_reference", "approach", "grasp_axis_points"} <= set(
            next_state_schema["required"]
        )
        return json.dumps({
            "group_id": -2,
            "reason": "no trajectory performs the insertion",
            "next_state": {
                "target_pixel": [130, 140],
                "orientation_reference": "image",
                "approach": "top_down",
                "grasp_axis_points": [[80, 120], [180, 120]],
                "gripper_position": 0.3,
            },
        })
    monkeypatch.setattr(vlm, "generate", generate)
    result = vlm.select_trajectory_group(
        "task", "pick", [{"group_id": 0, "atomic_action": "insert"}], image,
    )
    assert result["next_state"] == {
        "target_pixel": [130, 140],
        "orientation_reference": "image",
        "approach": "top_down",
        "grasp_axis_points": [[80, 120], [180, 120]],
        "gripper_position": 0.3,
    }
    monkeypatch.setattr(vlm, "generate", lambda *args: json.dumps({
        "group_id": -2,
        "reason": "missing target point",
        "next_state": {
            "orientation_reference": "image",
            "approach": "top_down",
            "grasp_axis_points": [[100, 128], [156, 128]],
            "gripper_position": 0.1,
        },
    }))
    with pytest.raises(ValueError, match="exactly one target_pixel"):
        vlm.select_trajectory_group(
            "task", "lift the object", [{"group_id": 0, "atomic_action": "lift_up"}], image,
        )
    monkeypatch.setattr(
        vlm, "generate", lambda *args: '{"group_id": -2, "reason": "missing pose"}',
    )
    with pytest.raises(ValueError, match="missing next_state"):
        vlm.select_trajectory_group("task", "pick", [{"group_id": 0}], image)
    legacy_state = {
        "group_id": -2,
        "reason": "legacy orientation",
        "next_state": {
            "target_pixel": [128, 128],
            "orientation_mode": "keep_current",
            "gripper_position": 0.0,
        },
    }
    monkeypatch.setattr(vlm, "generate", lambda *args: json.dumps(legacy_state))
    with pytest.raises(ValueError, match="orientation reference"):
        vlm.select_trajectory_group("task", "pick", [{"group_id": 0}], image)
    invalid_state = {
        "group_id": -2,
        "reason": "invalid gripper target",
        "next_state": {
            "target_pixel": [128, 128],
            "orientation_reference": "image",
            "approach": "top_down",
            "grasp_axis_points": [[100, 128], [156, 128]],
            "gripper_position": 1.2,
        },
    }
    monkeypatch.setattr(vlm, "generate", lambda *args: json.dumps(invalid_state))
    with pytest.raises(ValueError, match="gripper_position"):
        vlm.select_trajectory_group("task", "pick", [{"group_id": 0}], image)


@pytest.mark.skip(reason="legacy IK/NEXT_STATE protocol removed")
def test_google_group_selection_disables_next_state_when_near(monkeypatch):
    vlm = GoogleVLM(api_key="test")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    groups = [{
        "group_id": 0,
        "distance_class": "near",
        "target_distance_m": 0.12,
        "next_state_allowed": False,
    }]

    def choose_group(system, text, *args):
        assert "NEXT_STATE is unavailable" in text
        assert "group_id -2" in text
        assert "enum" not in args[-1]["properties"]["group_id"]
        assert "next_state" not in args[-1]["properties"]
        return '{"group_id": 0, "reason": "use a near-field VLA trajectory"}'

    monkeypatch.setattr(vlm, "generate", choose_group)
    assert vlm.select_trajectory_group("task", "pick", groups, image)["group_id"] == 0
    monkeypatch.setattr(
        vlm, "generate", lambda *args: '{"group_id": -2, "reason": "invalid near IK"}',
    )
    with pytest.raises(ValueError, match="invalid trajectory group"):
        vlm.select_trajectory_group("task", "pick", groups, image)


@pytest.mark.skip(reason="IK/NEXT_STATE removed")
def test_vlm_next_state_is_disabled_without_target_point(monkeypatch, tmp_path):
    class ChoosingVLM(PassthroughVLM):
        def select_trajectory_group(self, *args, **kwargs):
            groups = args[2]
            assert groups[0]["distance_class"] == "unavailable"
            assert groups[0]["next_state_allowed"] is False
            return {"group_id": 0, "reason": "use VLA for targetless lift"}
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=PassthroughStrategy(StrategyContext(vlm=ChoosingVLM())),
    ))
    monkeypatch.setattr(proxy, "_project_candidate_trajectories",
                        lambda chunks, *args: [{"left": [[0, 0], [10, 10]], "right": [[0, 0], [20, 20]]}
                                               for _ in chunks])
    state = SessionState(
        subgoals=["lift the held object"], subgoal_arms=["left"],
        subgoal_atomic_actions=["lift_up"], subgoal_coordinates=[[]],
        episode_log_dir=str(tmp_path),
    )
    obs = _observation(0)
    obs["observation/joint_position"] = np.zeros(12)
    obs["observation/gripper_position"] = np.array([0.2, 0.7])
    response = proxy._select_grouped_vla_response([{"actions": np.zeros((50, 14))}], obs, state)
    np.testing.assert_array_equal(response["actions"], np.zeros((30, 14)))
    selection = response["video_overlay"]["trajectory_selection"]
    assert selection["next_state_allowed"] is False
    assert selection["distance_class"] == "unavailable"


@pytest.mark.skip(reason="IK/NEXT_STATE removed")
def test_vlm_next_state_is_rejected_when_ee_is_near_target(monkeypatch, tmp_path):
    class ChoosingVLM(PassthroughVLM):
        def select_trajectory_group(self, instruction, subgoal, groups, *args, **kwargs):
            assert groups[0]["current_ee"]["gripper_position"] == pytest.approx(0.85)
            assert groups[0]["distance_class"] == "unavailable"
            assert groups[0]["next_state_allowed"] is False
            return {
                "group_id": -2,
                "reason": "the next state keeps the EE pose and closes the gripper",
                "next_state": {
                    "target_pixel": [100, 128],
                    "orientation_reference": "image",
                    "approach": "top_down",
                    "grasp_axis_points": [[100, 128], [156, 128]],
                    "gripper_position": 0.0,
                },
            }
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=PassthroughStrategy(StrategyContext(vlm=ChoosingVLM())),
    ))
    monkeypatch.setattr(
        proxy, "_project_candidate_trajectories",
        lambda chunks, *args: [
            {"left": [[0, 0], [10, 10]], "right": [[0, 0], [20, 20]]}
            for _ in chunks
        ],
    )
    current_ee_pose = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr("vlm_orchestrator.proxy.current_ee_pose",
                        lambda *args: (current_ee_pose, np.eye(3)))
    monkeypatch.setattr(
        "vlm_orchestrator.proxy.project_points",
        lambda *args: np.array([[100.0 * 7.0 / 255.0, 128.0 * 7.0 / 255.0]]),
    )
    state = SessionState(
        subgoals=["secure and move the object"], subgoal_arms=["right"],
        subgoal_atomic_actions=["pick"], subgoal_coordinates=[[[100, 128]]],
        episode_log_dir=str(tmp_path),
    )
    obs = _observation(0)
    obs["observation/joint_position"] = np.arange(12, dtype=float) * 0.01
    obs["observation/gripper_position"] = np.array([0.25, 0.85])
    obs["observation/camera_K"] = np.eye(3)
    obs["observation/camera_extrinsic"] = np.eye(4)

    with pytest.raises(ValueError, match="Invalid VLM trajectory group: -2"):
        proxy._select_grouped_vla_response(
            [{"actions": np.zeros((50, 14))}], obs, state,
        )


@pytest.mark.skip(reason="IK/NEXT_STATE removed")
def test_vlm_can_request_ik_when_all_groups_are_wrong(monkeypatch, tmp_path):
    class ChoosingVLM(PassthroughVLM):
        def select_trajectory_group(self, instruction, subgoal, groups, image, extra_images=None, *, before_images=None, previous_available=False, memory="", last_executed_ee_trajectory=None, **kwargs):
            assert groups[0]["distance_class"] == "far"
            assert groups[0]["next_state_allowed"] is True
            return {
                "group_id": -2,
                "reason": "all trajectories move away from target",
                "next_state": {
                    "target_pixel": [10, 10],
                    "orientation_reference": "image",
                    "approach": "top_down",
                    "grasp_axis_points": [[5, 10], [20, 10]],
                    "gripper_position": 0.0,
                },
            }
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=PassthroughStrategy(StrategyContext(vlm=ChoosingVLM())),
    ))
    monkeypatch.setattr(proxy, "_project_candidate_trajectories",
                        lambda chunks, *args: [{"left": [[0, 0], [10, 10]], "right": [[0, 0], [20, 20]]}
                                               for _ in chunks])
    monkeypatch.setattr("vlm_orchestrator.proxy.target_pixel_to_world",
                        lambda *args: np.array([0.1, 0.2, 0.3]))
    monkeypatch.setattr("vlm_orchestrator.proxy.target_gripper_rotation",
                        lambda *args: np.eye(3))
    monkeypatch.setattr("vlm_orchestrator.proxy.project_points",
                        lambda *args: np.array([[7.0, 7.0]]))
    planned_position = np.array([0.05, 0.1, 0.2])
    planned_rotation = np.diag([1.0, -1.0, -1.0])
    mount = np.eye(4)
    def center_wrist(joints, arm, target, wrist_extrinsic, preferred_rotation):
        assert arm == "left"
        np.testing.assert_array_equal(target, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(wrist_extrinsic, np.eye(4))
        np.testing.assert_array_equal(preferred_rotation, np.eye(3))
        return planned_position, planned_rotation, {"ee_to_wrist_camera": mount.tolist()}
    monkeypatch.setattr("vlm_orchestrator.proxy.wrist_centered_ee_target", center_wrist)
    ik_actions = np.full((10, 14), 7.0)
    def solve_ik(*args, **kwargs):
        np.testing.assert_array_equal(args[3], planned_position)
        np.testing.assert_array_equal(kwargs["target_rotation"], planned_rotation)
        assert kwargs["grasp_axis_symmetric"] is False
        assert kwargs["max_joint_delta"] == pytest.approx(np.pi)
        assert kwargs["tolerance_m"] == pytest.approx(0.002)
        assert kwargs["orientation_tolerance_rad"] == pytest.approx(0.01)
        assert kwargs["steps"] == 10
        return ik_actions.copy(), {"position_error_m": 0.005, "target_reached": True}
    monkeypatch.setattr("vlm_orchestrator.proxy.ik_approach_chunk", solve_ik)
    def validate_wrist(joints, arm, target, ee_to_camera, intrinsic, image_shape):
        np.testing.assert_array_equal(joints, ik_actions[-1])
        np.testing.assert_array_equal(ee_to_camera, mount)
        np.testing.assert_array_equal(intrinsic, np.eye(3))
        assert image_shape == (8, 8)
        return {
            "wrist_target_visible": True,
            "wrist_target_reached": True,
            "wrist_center_error_px": 0.5,
            "wrist_target_distance_m": 0.15,
        }
    monkeypatch.setattr("vlm_orchestrator.proxy.evaluate_wrist_target", validate_wrist)
    state = SessionState(subgoals=["pick"], subgoal_arms=["left"],
                         subgoal_atomic_actions=["pick"], subgoal_coordinates=[[[10, 10]]],
                         episode_log_dir=str(tmp_path))
    obs = _observation(0)
    obs.update({
        "observation/joint_position": np.zeros(12),
        "observation/gripper_position": np.zeros(2),
        "observation/depth_exterior_image_1_left": np.ones((8, 8)),
        "observation/camera_K": np.eye(3),
        "observation/camera_extrinsic": np.eye(4),
        "observation/wrist_camera_K_left": np.eye(3),
        "observation/wrist_camera_extrinsic_left": np.eye(4),
    })
    response = proxy._select_grouped_vla_response(
        [{"actions": np.zeros((50, 14))}], obs, state,
    )
    expected = ik_actions.copy()
    expected[:, 6] = 0.0
    np.testing.assert_array_equal(response["actions"], expected)
    selection = response["video_overlay"]["trajectory_selection"]
    assert selection["selection_reason"] == "vlm_next_state"
    assert selection["selected_group_id"] == -2
    assert selection["next_state_selected"] is True
    assert selection["ik_diagnostics"]["position_error_m"] == 0.005
    assert selection["next_state"]["approach"] == "top_down"
    assert selection["execution_steps"] == 10
    assert response["orchestrator_action_chunk_size"] == 10

    monkeypatch.setattr("vlm_orchestrator.proxy.evaluate_wrist_target", lambda *args: {
        "wrist_target_visible": True,
        "wrist_target_reached": False,
        "wrist_center_error_px": 12.0,
        "wrist_target_distance_m": 0.15,
    })
    with pytest.raises(ValueError, match="did not satisfy.*center_error=12.00px"):
        proxy._select_grouped_vla_response(
            [{"actions": np.zeros((50, 14))}], obs, state,
        )

    del obs["observation/wrist_camera_K_left"]
    with pytest.raises(ValueError, match="requires left wrist image, intrinsics, and extrinsics"):
        proxy._select_grouped_vla_response(
            [{"actions": np.zeros((50, 14))}], obs, state,
        )


def test_grouping_uses_ee_and_executes_one_group_member(monkeypatch):
    class ChoosingVLM(PassthroughVLM):
        def select_trajectory_group(self, instruction, subgoal, groups, image, extra_images=None, *, before_images=None, previous_available=False, memory="", last_executed_ee_trajectory=None, **kwargs):
            assert all("candidate_indices" not in group for group in groups)
            assert groups[0]["endpoint_space"] == "world_xyz_meters"
            assert groups[0]["trajectory_steps"] == 10
            assert groups[0]["overlay_trajectory_steps"] == 30
            assert groups[0]["endpoint_radius_m"] == pytest.approx(0.05)
            assert "mean_ee_endpoints" not in groups[0]
            assert "mean_ee_trajectory_z_ranges_m" not in groups[0]
            assert set(groups[0]["mean_ee_trajectory"]) == {"left", "right"}
            return {"group_id": 0, "reason": "correct height"}
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=PassthroughStrategy(StrategyContext(vlm=ChoosingVLM())),
    ))
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate",
                        lambda chunks, **kwargs: (0, {"trajectory_steps_requested": 30}))
    def fk(chunk):
        values = np.asarray(chunk)
        value = float(values[-1, 0])
        rise = len(values) * 0.001
        return {
            "left": np.array([[0, 0, value * 0.1 + rise], [0, 0, value * 0.1]]),
            "right": np.array([[0, 0, rise], [0, 0, 0]]),
        }
    monkeypatch.setattr("vlm_orchestrator.proxy.fk_trajectory", fk)
    projected_inputs = []
    def project(chunks, *args):
        projected_inputs.append([np.asarray(chunk).copy() for chunk in chunks])
        return [{"left": [[1, 1], [10, 10]], "right": [[1, 1], [10, 10]]}
                for _ in chunks]
    monkeypatch.setattr(proxy, "_project_candidate_trajectories", project)
    state = SessionState(subgoals=["pick"], subgoal_arms=["left"],
                         subgoal_atomic_actions=["pick"], subgoal_coordinates=[[[10, 10]]])
    chunks = [np.full((50, 14), value) for value in [0, 0.01, 0.02, 0.8]]
    chunks[1][:, 6] = 0.3
    response = proxy._select_grouped_vla_response(
        [{"actions": chunk} for chunk in chunks],
        _observation(0), state,
    )
    selection = response["video_overlay"]["trajectory_selection"]
    assert selection["consistent_candidate_indices"] == [0, 2]
    assert selection["selected_candidate_index"] is None
    assert selection["averaged_candidate_indices"] == [0, 2]
    np.testing.assert_allclose(response["actions"], np.full((50, 14), 0.01))
    # Projection input is one 30-step mean action per group.
    assert len(projected_inputs[0]) == len(selection["groups"])


class StubConnection:
    def __init__(self):
        self.calls = []

    async def recv_metadata(self):
        return {}

    async def infer(self, observation):
        return {"actions": np.zeros((2, 14))}

    async def infer_batch(self, observations):
        self.calls.append(observations)
        return [{"actions": np.full((2, 14), index)} for index in range(len(observations))]

    async def close(self):
        return None


class StubBackend:
    def __init__(self):
        self.connection = StubConnection()

    async def connect(self):
        return self.connection


class StubSession:
    def __init__(self, observations):
        self.observations = [observations, None]
        self.sent = []

    async def send_metadata(self, metadata):
        return None

    async def recv_batch(self):
        return self.observations.pop(0)

    async def send_action_batch(self, actions):
        self.sent.append(actions)
        return [len(action["actions"]) for action in actions]


def _observation(env_idx):
    return {
        "env_idx": env_idx,
        "prompt": "move",
        "observation/exterior_image_1_left": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def test_records_only_the_confirmed_executed_chunk_ee_trajectory(monkeypatch):
    def fake_fk(steps):
        count = len(steps)
        positions = np.arange(count, dtype=float)
        return {
            "left": np.column_stack((positions, np.zeros(count), positions * 0.01)),
            "right": np.column_stack((np.zeros(count), positions, positions * -0.01)),
        }

    monkeypatch.setattr("vlm_orchestrator.proxy.fk_trajectory", fake_fk)
    actions = np.zeros((12, 14), dtype=float)
    actions[:, 6] = np.arange(12) * 0.1
    actions[:, 13] = np.arange(12) * 0.2
    response = {
        "actions": actions,
        "video_overlay": {"trajectory_selection": {
            "target_arm": "left",
            "atomic_action": "pick",
            "target_points": [[80, 120]],
        }},
    }
    first = SessionState(target_revision=4, subgoals=["Move the target."])
    second = SessionState()

    OrchestratorProxy._record_executed_chunk(first, response, sent_count=10)

    trajectory = first.last_executed_ee_trajectory
    assert trajectory["executed_steps"] == 10
    assert trajectory["atomic_action"] == "pick"
    assert trajectory["target_points"] == [[80, 120]]
    assert trajectory["coordinate_frame"] == "world_m"
    assert [step["step_index"] for step in trajectory["steps"]] == list(range(10))
    assert trajectory["steps"][9]["left"]["position_world_m"] == [9.0, 0.0, 0.09]
    assert trajectory["steps"][9]["left"]["gripper_position"] == pytest.approx(0.9)
    assert trajectory["steps"][9]["right"]["gripper_position"] == pytest.approx(1.8)
    assert not ({"delta_world_m", "path_length_m", "min_z_m", "max_z_m"} & set(trajectory))
    assert first.log_entries[-1]["type"] == "executed_action_ee_trajectory"
    assert first.log_entries[-1]["trajectory"] == trajectory
    OrchestratorProxy._record_executed_chunk(second, response, sent_count=3)
    assert len(second.last_executed_ee_trajectory["steps"]) == 3
    assert len(first.last_executed_ee_trajectory["steps"]) == 10


def test_recent_vlm_ee_window_combines_only_dispatched_chunks(monkeypatch):
    def fake_fk(steps):
        count = len(steps)
        positions = np.arange(count, dtype=float)
        return {
            "left": np.column_stack((positions, np.zeros(count), positions * 0.01)),
            "right": np.column_stack((np.zeros(count), positions, positions * -0.01)),
        }

    monkeypatch.setattr("vlm_orchestrator.proxy.fk_trajectory", fake_fk)
    response = {
        "actions": np.zeros((50, 14), dtype=float),
        "video_overlay": {"trajectory_selection": {
            "atomic_action": "press",
            "target_points": [[128, 142]],
        }},
    }
    state = SessionState()
    OrchestratorProxy._record_executed_chunk(state, response, sent_count=10)
    OrchestratorProxy._record_executed_chunk(state, response, sent_count=10)

    recent = state.recent_executed_ee_trajectory()
    assert recent["executed_steps"] == 20
    assert recent["window_steps"] == 20
    assert len(recent["steps"]) == 20
    assert recent["chunk_boundaries"] == [
        {
            "chunk_index": 0,
            "label": "previous_executed_chunk",
            "start_step": 0,
            "end_step": 9,
            "executed_steps": 10,
        },
        {
            "chunk_index": 1,
            "label": "latest_executed_chunk",
            "start_step": 10,
            "end_step": 19,
            "executed_steps": 10,
        },
    ]
    assert [step["step_index"] for step in recent["steps"]] == list(range(20))
    assert all(step["chunk_step_index"] < 10 for step in recent["steps"])
    assert recent["steps"][0]["execution_chunk"] == "previous_executed_chunk"
    assert recent["steps"][-1]["execution_chunk"] == "latest_executed_chunk"


def test_pi05_receives_high_level_instruction_not_subgoal():
    state = SessionState(
        original_instruction="Cover the blocks, then uncover them by color.",
        vla_prompt="Pick the left cup at [80, 120].",
        subgoals=["Pick the left cup at [80, 120]."],
        subgoal_coordinates=[[[80, 120]]],
        subgoal_atomic_actions=["pick"],
        subgoal_arms=["left"],
    )
    obs = _observation(0)
    obs["prompt"] = obs["instruction"] = state.vla_prompt
    prepared = OrchestratorProxy._prepare_vla_observation(obs, state, None, None, False)
    assert prepared["instruction"] == state.original_instruction
    assert prepared["prompt"] == state.original_instruction
    assert prepared["coordinates"] == [[80, 120]]
    assert prepared["atomic_action"] == "pick"
    assert "arm" not in prepared


def test_completed_plan_removes_final_subgoal_guidance():
    state = SessionState(
        original_instruction="Press the buttons, then return to origin.",
        subgoals=["Press the blue button at [158, 142]."],
        subgoal_coordinates=[[[158, 142]]],
        subgoal_atomic_actions=["press_down"],
        subgoal_arms=["right"],
        task_completed=True,
    )
    obs = _observation(0)
    obs["coordinates"] = [[158, 142]]
    obs["atomic_action"] = "press_down"

    prepared = OrchestratorProxy._prepare_vla_observation(
        obs, state, None, None, False,
    )
    metadata = OrchestratorProxy._attach_metadata(
        {"actions": np.zeros((1, 14))}, obs, state,
    )

    assert prepared["instruction"] == state.original_instruction
    assert "coordinates" not in prepared
    assert "atomic_action" not in prepared
    assert "orchestrator_active_subgoal" not in metadata
    assert "atomic_action" not in metadata.get("video_overlay", {})


def _scored_selection(chunks, **kwargs):
    scores = [float(np.asarray(chunk)[0, 0]) for chunk in chunks]
    best = int(np.argmin(scores))
    return best, {"geometry_scores": scores, "candidate_pool": [best], "trajectory_steps_requested": 30}


def test_retains_old_remainder_unless_new_path_improves(monkeypatch):
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate", _scored_selection)
    proxy = OrchestratorProxy(ProxyConfig())
    old = np.zeros((50, 14))
    old[:, 0] = 10
    state = SessionState(action_cache=old, action_cache_offset=10, action_cache_target_revision=0)
    new = old.copy()
    new[:, 0] = 9.5
    response = proxy._select_vla_response([{"actions": new}], _observation(0), state)
    np.testing.assert_array_equal(response["actions"], old[10:])
    assert response["video_overlay"]["trajectory_selection"]["selection_reason"] == "retain_previous_trajectory"


def test_switches_to_materially_better_continuous_path(monkeypatch):
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate", _scored_selection)
    proxy = OrchestratorProxy(ProxyConfig())
    old = np.zeros((50, 14))
    old[:, 0] = 10
    state = SessionState(action_cache=old, action_cache_offset=10, action_cache_target_revision=0)
    new = old.copy()
    new[:, 0] = 0
    # Separate geometric score from joint continuity for this test.
    def score(chunks, **kwargs):
        scores = [0.0 if np.asarray(chunk)[0, 0] == 0 else 100.0 for chunk in chunks]
        return 0, {"geometry_scores": scores, "candidate_pool": [0]}
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate", score)
    response = proxy._select_vla_response([{"actions": new}], _observation(0), state)
    np.testing.assert_array_equal(response["actions"], new)
    assert response["video_overlay"]["trajectory_selection"]["selection_reason"] == "switch_improved_trajectory"


def test_target_revision_retains_previous_candidate(monkeypatch):
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate", _scored_selection)
    proxy = OrchestratorProxy(ProxyConfig())
    state = SessionState(action_cache=np.zeros((50, 14)), action_cache_offset=10,
                         action_cache_target_revision=0, target_revision=1)
    new = np.ones((50, 14))
    response = proxy._select_vla_response([{"actions": new}], _observation(0), state)
    np.testing.assert_array_equal(response["actions"], np.zeros((40, 14)))
    assert response["video_overlay"]["trajectory_selection"]["retained_candidate_index"] == 1


def test_short_old_remainder_is_retained(monkeypatch):
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate", _scored_selection)
    proxy = OrchestratorProxy(ProxyConfig())
    state = SessionState(action_cache=np.zeros((30, 14)), action_cache_offset=10,
                         action_cache_target_revision=0, target_revision=0)
    new = np.ones((50, 14))
    response = proxy._select_vla_response([{"actions": new}], _observation(0), state)
    np.testing.assert_array_equal(response["actions"], np.zeros((20, 14)))
    selection = response["video_overlay"]["trajectory_selection"]
    assert selection["retained_candidate_index"] == 1
    assert selection["selection_reason"] == "retain_previous_trajectory"


def test_rejects_average_that_loses_target(monkeypatch):
    def score(chunks, **kwargs):
        scores = [float(np.asarray(chunk)[0, 0]) for chunk in chunks]
        return 0, {"geometry_scores": scores, "candidate_pool": list(range(len(chunks)))}
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate", score)
    monkeypatch.setattr("vlm_orchestrator.proxy.average_most_similar_chunks",
                        lambda chunks, pool: (np.full((50, 14), 100.0), [0, 1], 0.0))
    proxy = OrchestratorProxy(ProxyConfig())
    response = proxy._select_vla_response(
        [{"actions": np.ones((50, 14))}, {"actions": np.full((50, 14), 1.05)}],
        _observation(0), SessionState(),
    )
    np.testing.assert_array_equal(response["actions"], np.ones((50, 14)))
    assert response["video_overlay"]["trajectory_selection"]["average_rejected"] is True


def test_overlay_projects_actual_average_and_execution_window(monkeypatch):
    proxy = OrchestratorProxy(ProxyConfig())
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate",
                        lambda chunks, **kwargs: (0, {"candidate_pool": [0, 1], "trajectory_steps_requested": 30}))
    calls = []
    def project(chunks, intrinsic, extrinsic, shape, steps=None):
        calls.append((chunks, steps))
        return [{"left": [[0, 0], [1, 1]], "right": []} for chunk in chunks]
    monkeypatch.setattr(proxy, "_project_candidate_trajectories", project)
    response = proxy._select_vla_response(
        [{"actions": np.zeros((50, 14))}, {"actions": np.ones((50, 14))}],
        _observation(0), SessionState(),
    )
    np.testing.assert_array_equal(calls[-1][0][0], response["actions"])
    assert calls[-1][1] == 30
    selection = response["video_overlay"]["trajectory_selection"]
    assert selection["executed_trajectory_index"] == 2
    assert selection["execution_steps"] == 10


def test_proxy_expands_candidates_and_preserves_environment_order():
    backend = StubBackend()
    session = StubSession([_observation(4), _observation(9)])
    strategy = PassthroughStrategy(StrategyContext(vlm=PassthroughVLM()))
    proxy = OrchestratorProxy(ProxyConfig(strategy=strategy, backend=backend, vla_candidates=3, action_chunk_size=1))
    asyncio.run(proxy._handle_session(session))
    assert len(backend.connection.calls) == 1
    assert len(backend.connection.calls[0]) == 6
    assert len(session.sent) == 1
    assert len(session.sent[0]) == 2
    assert all(len(item["actions"]) == 1 for item in session.sent[0])


def test_proxy_reserves_one_candidate_slot_for_eligible_previous():
    class LongChunkConnection(StubConnection):
        async def infer_batch(self, observations):
            self.calls.append(observations)
            return [{"actions": np.full((50, 14), index)} for index in range(len(observations))]

    backend = StubBackend()
    backend.connection = LongChunkConnection()
    observation = _observation(0)
    session = StubSession([observation])
    session.observations = [[observation], [observation], None]
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=PassthroughStrategy(StrategyContext(vlm=PassthroughVLM())),
        backend=backend,
        vla_candidates=16,
        action_chunk_size=10,
    ))

    asyncio.run(proxy._handle_session(session))

    assert [len(call) for call in backend.connection.calls] == [16, 15]
    second_selection = session.sent[1][0]["video_overlay"]["trajectory_selection"]
    assert second_selection["retained_candidate_index"] == 15


def test_combined_group_advance_discards_old_candidates_and_reinfers(monkeypatch):
    class CombinedVLM(PassthroughVLM):
        def __init__(self):
            self.selected_subgoals = []
            self.selection_only_calls = []

        def decompose(self, instruction, image, extra_images=None):
            return DecompositionResult(
                subgoals=["Press left.", "Press blue."],
                atomic_actions=["press", "press"],
                coordinates=[[[97, 142]], [[158, 142]]],
                arms=["left", "right"],
            )

        def assess_progress(self, *args, **kwargs):
            pytest.fail("Non-observe progress must be merged into Group selection")

        def select_trajectory_group(
            self, instruction, subgoal, groups, image, extra_images=None, *,
            before_images=None, previous_available=False, memory="",
            last_executed_ee_trajectory=None, **kwargs,
        ):
            self.selected_subgoals.append(subgoal)
            self.selection_only_calls.append(bool(kwargs.get("selection_only")))
            advance = len(self.selected_subgoals) == 1
            if kwargs.get("selection_only"):
                return {"group_id": 0, "reason": "selection-only new target"}
            return {
                "group_id": 0,
                "reason": "best group for displayed subgoal",
                "decision": "advance" if advance else "stay",
                "progress_reason": "left press is complete" if advance else "blue press remains",
                "memory": "left press completed",
                "before_to_now_summary": "The first subgoal completed.",
                "replan": False,
            }

    class RetryConnection(StubConnection):
        async def infer_batch(self, observations):
            self.calls.append(observations)
            value = float(len(self.calls))
            return [{"actions": np.full((2, 14), value)} for _ in observations]

    backend = StubBackend()
    backend.connection = RetryConnection()
    vlm = CombinedVLM()
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm),
            SubgoalConfig(progress_interval=1),
    )
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=strategy, backend=backend, vla_candidates=1, action_chunk_size=1,
    ))
    monkeypatch.setattr(
        proxy, "_project_candidate_trajectories",
        lambda chunks, *args: [
            {"left": [[0, 0], [10, 10]], "right": [[0, 0], [20, 20]]}
            for _ in chunks
        ],
    )
    session = StubSession([_observation(0)])

    asyncio.run(proxy._handle_session(session))

    assert vlm.selected_subgoals == ["Press left.", "Press blue."]
    assert vlm.selection_only_calls == [False, True]
    assert len(backend.connection.calls) == 2
    np.testing.assert_array_equal(session.sent[0][0]["actions"], np.full((1, 14), 2.0))
    assert session.sent[0][0]["orchestrator_active_subgoal"] == "Press blue."
    selection = session.sent[0][0]["video_overlay"]["trajectory_selection"]
    assert selection["selection_only"] is True
    assert selection["vlm_replan"] is False
    assert selection["progress_decision"] == "advance"
    assert selection["progress_reason"] == "left press is complete"
    assert selection["progress_memory"] == "left press completed"
    assert selection["before_to_now_summary"] == "The first subgoal completed."


def test_final_group_advance_executes_selected_cleanup_actions(monkeypatch):
    class FinalVLM(PassthroughVLM):
        def decompose(self, instruction, image, extra_images=None):
            return DecompositionResult(
                subgoals=["Press the final button."],
                atomic_actions=["press_down"],
                coordinates=[[[158, 142]]],
                arms=["right"],
            )

        def select_trajectory_group(self, *args, **kwargs):
            return {
                "group_id": 0,
                "reason": "The final press is complete; execute the cleanup tail.",
                "decision": "advance",
                "progress_reason": "The final button was pressed.",
                "memory": "All requested presses are complete.",
                "before_to_now_summary": "The final button moved down.",
                "replan": False,
            }

    cleanup_actions = np.full((2, 14), 7.0)

    class CleanupConnection(StubConnection):
        async def infer_batch(self, observations):
            self.calls.append(observations)
            return [{"actions": cleanup_actions.copy()} for _ in observations]

    backend = StubBackend()
    backend.connection = CleanupConnection()
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=SubgoalStrategy(StrategyContext(vlm=FinalVLM())),
        backend=backend,
        vla_candidates=1,
        action_chunk_size=1,
    ))
    monkeypatch.setattr(
        proxy, "_project_candidate_trajectories",
        lambda chunks, *args: [
            {"left": [[0, 0]], "right": [[158, 142]]} for _ in chunks
        ],
    )
    session = StubSession([_observation(0)])

    asyncio.run(proxy._handle_session(session))

    np.testing.assert_array_equal(session.sent[0][0]["actions"], cleanup_actions[:1])
    assert "orchestrator_active_subgoal" not in session.sent[0][0]


@pytest.mark.skip(reason="IK/NEXT_STATE removed")
def test_session_sends_all_next_state_actions(monkeypatch):
    class PickStrategy(PassthroughStrategy):
        def process(self, obs, state):
            state.subgoals = ["move to next state"]
            state.subgoal_atomic_actions = ["move"]
            state.subgoal_arms = ["left"]
            state.subgoal_coordinates = [[]]
            return obs, state

    backend = StubBackend()
    session = StubSession([_observation(0)])
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=PickStrategy(StrategyContext(vlm=PassthroughVLM())),
        backend=backend, action_chunk_size=10,
    ))
    next_state_actions = np.arange(10 * 14, dtype=float).reshape(10, 14)
    monkeypatch.setattr(proxy, "_select_grouped_vla_response", lambda *args: {
        "actions": next_state_actions,
        "orchestrator_action_chunk_size": 10,
        "video_overlay": {"trajectory_selection": {"next_state_selected": True}},
    })

    asyncio.run(proxy._handle_session(session))

    response = session.sent[0][0]
    np.testing.assert_array_equal(response["actions"], next_state_actions)
    assert response["orchestrator_action_cache_offset"] == [0, 10]


def test_session_routes_pick_through_vlm_groups(monkeypatch):
    class ChoosingVLM(PassthroughVLM):
        def select_trajectory_group(self, instruction, subgoal, groups, image, extra_images=None, *, before_images=None, previous_available=False, memory="", last_executed_ee_trajectory=None, **kwargs):
            return {"group_id": 1, "reason": "right endpoint"}
    class PickStrategy(PassthroughStrategy):
        def process(self, obs, state):
            state.original_instruction = "move"
            state.subgoals = ["pick"]
            state.subgoal_atomic_actions = ["pick"]
            state.subgoal_arms = ["left"]
            state.subgoal_coordinates = [[[100, 20]]]
            return obs, state
    backend = StubBackend()
    session = StubSession([_observation(0)])
    proxy = OrchestratorProxy(ProxyConfig(
        strategy=PickStrategy(StrategyContext(vlm=ChoosingVLM())),
        backend=backend, vla_candidates=3, action_chunk_size=1,
    ))
    monkeypatch.setattr("vlm_orchestrator.proxy.select_candidate",
                        lambda chunks, **kwargs: (0, {"trajectory_steps_requested": 10}))
    monkeypatch.setattr(proxy, "_project_candidate_trajectories",
                        lambda chunks, *args: [{"left": [[0, 0], [float(np.asarray(chunk)[0, 0]) * 100, 0]],
                                                "right": [[0, 0], [0, 0]]} for chunk in chunks])
    monkeypatch.setattr(proxy, "_select_vla_response",
                        lambda *args: pytest.fail("Legacy selection must not override VLM group choice"))
    asyncio.run(proxy._handle_session(session))
    response = session.sent[0][0]
    np.testing.assert_array_equal(response["actions"], np.ones((1, 14)))
    assert response["video_overlay"]["trajectory_selection"]["selected_group_id"] == 1


def test_proxy_saves_projected_overlay_image(tmp_path):
    state = SessionState(episode_log_dir=str(tmp_path))
    overlay = {
        "trajectory_selection": {
            "projected_trajectories": [{"left": [[0, 0], [255, 255]], "right": []}],
            "target_points": [[128, 64]],
        }
    }
    OrchestratorProxy._save_overlay_image(
        _observation(0), state, overlay, selected=0
    )
    path = tmp_path / "vla_overlay_000000.png"
    assert path.exists()
    assert overlay["trajectory_selection"]["overlay_image_path"] == str(path)


def test_group_selection_overlay_includes_current_memory(tmp_path):
    state = SessionState(episode_log_dir=str(tmp_path))
    panels = [np.zeros((20, 30, 3), dtype=np.uint8)]
    without_memory = OrchestratorProxy._save_group_selection_image(
        state, panels, 0, "select the target", ["G0"]
    )
    first_height = Image.open(without_memory).height
    state.infer_count = 1
    with_memory = OrchestratorProxy._save_group_selection_image(
        state, panels, 0, "select the target", ["G0"],
        "The left button is already pressed; continue with the next subtask.",
        "The gripper moved down from the previous view and contact is now visible.",
    )
    assert without_memory is not None and with_memory is not None
    assert first_height < Image.open(with_memory).height


def test_group_selection_overlay_includes_before_to_now_summary(tmp_path):
    state = SessionState(episode_log_dir=str(tmp_path))
    panel = [np.zeros((20, 30, 3), dtype=np.uint8)]
    path = OrchestratorProxy._save_group_selection_image(
        state, panel, 0, "select the target", ["G0"], "integrated memory",
        "The left arm descended and touched the button.",
    )
    assert path is not None
    with Image.open(path) as image:
        assert image.height > 20 + 10 + 18 * 2


def test_proxy_overlay_draws_both_arms(tmp_path):
    state = SessionState(episode_log_dir=str(tmp_path))
    observation = _observation(0)
    observation["observation/exterior_image_1_left"] = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay = {
        "trajectory_selection": {
            "projected_trajectories": [
                {"left": [[40, 160], [80, 160]], "right": [[160, 160], [220, 160]]}
            ],
            "target_points": [],
            "target_arm": "left",
        }
    }

    OrchestratorProxy._save_overlay_image(observation, state, overlay, selected=0)

    rendered = np.asarray(Image.open(tmp_path / "vla_overlay_000000.png"))
    assert rendered[40, 18].max() > 0
    assert rendered[40, 50].max() > 0


def test_proxy_projects_all_candidates_for_overlay_regardless_of_selection_mode():
    chunks = [np.zeros((2, 14)), np.full((2, 14), 0.1)]
    intrinsic = np.asarray([[100, 0, 112], [0, 100, 112], [0, 0, 1]])
    camera_to_world = np.diag([1.0, 1.0, -1.0, 1.0])

    projected = OrchestratorProxy._project_candidate_trajectories(
        chunks, intrinsic, camera_to_world, (224, 224)
    )

    assert len(projected) == 2
    assert all(set(candidate) == {"left", "right"} for candidate in projected)
    assert all(len(candidate[arm]) == 2 for candidate in projected for arm in candidate)


def test_proxy_overlay_uses_requested_trajectory_window():
    chunks = [np.zeros((40, 14)), np.full((40, 14), 0.1)]
    intrinsic = np.asarray([[100, 0, 112], [0, 100, 112], [0, 0, 1]])
    camera_to_world = np.diag([1.0, 1.0, -1.0, 1.0])

    projected = OrchestratorProxy._project_candidate_trajectories(
        chunks, intrinsic, camera_to_world, (224, 224), trajectory_steps=10
    )

    assert all(len(candidate[arm]) == 10 for candidate in projected for arm in candidate)


def test_proxy_observe_selects_candidate_closest_to_dual_arm_zero_pose():
    backend = StubBackend()
    strategy = PassthroughStrategy(StrategyContext(vlm=PassthroughVLM()))
    proxy = OrchestratorProxy(ProxyConfig(strategy=strategy, backend=backend))
    state = SessionState(
        subgoals=["Observe the target."],
        subgoal_atomic_actions=["observe"],
        subgoal_coordinates=[[[128, 128]]],
        subgoal_arms=["left"],
    )
    far = np.full((2, 14), 0.5)
    near = np.full((2, 14), 0.05)
    far[:, [6, 13]] = 0.0
    near[:, [6, 13]] = 1000.0

    selected = proxy._select_vla_response(
        [{"actions": far}, {"actions": near}], _observation(0), state
    )

    np.testing.assert_allclose(selected["actions"], near)
    diagnostics = selected["video_overlay"]["trajectory_selection"]
    assert diagnostics["selected_index"] == 1
    assert diagnostics["selection_reason"] == "observe_zero_pose"
    assert diagnostics["averaged_candidate_indices"] == [1]


def test_cache_refreshes_after_selected_chunk_is_exhausted():
    state = SessionState(target_revision=0)
    first = {"actions": np.zeros((1, 14))}
    second = {"actions": np.ones((1, 14))}
    assert OrchestratorProxy._cached_action_chunk(first, state, 1)["actions"][0][0] == 0
    refreshed = OrchestratorProxy._cached_action_chunk(second, state, 1)
    assert refreshed["actions"][0][0] == 1
    assert refreshed["orchestrator_action_cache_refreshed"] is True


def test_google_group_selection_requires_best_available_group(monkeypatch):
    vlm = GoogleVLM(api_key="test")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    captured = {}

    def generate(system, text, *args):
        captured["text"] = text
        captured["schema"] = args[-1]
        return (
            '{"group_id":0,"reason":"best available","decision":"stay",'
            '"candidate_index":0,"progress_reason":"not complete","memory":"still approaching",'
            '"before_to_now_summary":"The left gripper descended toward the target.",'
            '"replan":true}'
        )

    monkeypatch.setattr(vlm, "generate", generate)
    result = vlm.select_trajectory_group(
        "task", "pick", [{"group_id": 0, "atomic_action": "pick"}], image,
    )
    assert result["group_id"] == 0
    assert result["replan"] is True
    assert result["decision"] == "stay"
    assert result["memory"] == "still approaching"
    assert result["before_to_now_summary"] == "The left gripper descended toward the target."
    assert "Always select the best available group" in captured["text"]
    assert "-2" not in captured["text"]
    assert "-2" not in captured["schema"]["properties"]["group_id"]["description"]
    assert set(captured["schema"]["properties"]) == {
        "group_id", "reason", "decision", "progress_reason", "decision_reason", "memory",
        "before_to_now_summary", "replan",
    }
    assert "grasp_secure" not in captured["schema"]["properties"]
    assert "object_lifted" not in captured["schema"]["properties"]
    assert "next_state" not in result
    monkeypatch.setattr(vlm, "generate", lambda *args: '{"group_id": -2, "reason": "reject all"}')
    with pytest.raises(ValueError, match="invalid trajectory group"):
        vlm.select_trajectory_group("task", "pick", [{"group_id": 0}], image)


def test_google_press_description_uses_press_label(monkeypatch):
    vlm = GoogleVLM(api_key="test")
    captured = {}

    def generate(system, text, *args):
        captured["text"] = text
        return (
            '{"group_id":0,"reason":"presses","decision":"stay",'
            '"progress_reason":"incomplete","memory":"","before_to_now_summary":"",'
            '"replan":false}'
        )

    monkeypatch.setattr(vlm, "generate", generate)
    vlm.select_trajectory_group(
        "task", "Press the blue button.",
        [{"group_id": 0, "atomic_action": "press"}],
        np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert (
        "Current atomic action description: Close the gripper, press the designated control device "
        "to its lowest position, and then lift it slightly."
    ) in captured["text"]
    assert "Current atomic action: press" in captured["text"]


def test_google_selection_only_retry_has_no_progress_fields(monkeypatch):
    vlm = GoogleVLM(api_key="test")
    captured = {}

    def generate(system, text, image, extra_images, schema):
        captured["text"] = text
        captured["schema"] = schema
        return '{"group_id":0,"reason":"best group for the new target"}'

    monkeypatch.setattr(vlm, "generate", generate)
    result = vlm.select_trajectory_group(
        "task", "next subgoal", [{"group_id": 0}],
        np.zeros((8, 8, 3), dtype=np.uint8),
        selection_only=True,
    )

    assert set(captured["schema"]["properties"]) == {"group_id", "reason"}
    assert set(result) == {"group_id", "reason", "raw"}
    assert "Do not reassess progress" in captured["text"]
    assert "decision=advance" not in captured["text"]
    assert "Set replan=true" not in captured["text"]


def test_ten_environment_vlm_stages_are_concurrent(monkeypatch):
    """Barriers fail on serial decomposition or serial trajectory assessment."""
    import threading

    planning = threading.Barrier(10, timeout=10)
    selection = threading.Barrier(10, timeout=10)

    class ConcurrentStrategy(PassthroughStrategy):
        def process(self, obs, state):
            planning.wait()
            state.original_instruction = obs["prompt"]
            state.subgoals = ["Pick the block."]
            state.subgoal_atomic_actions = ["pick"]
            state.subgoal_arms = ["left"]
            return obs, state

    def select(candidates, obs, state, before_images, **kwargs):
        selection.wait()
        return {**candidates[0], "video_overlay": {"trajectory_selection": {}}}

    proxy = OrchestratorProxy(ProxyConfig(
        strategy=ConcurrentStrategy(StrategyContext(vlm=PassthroughVLM())),
        backend=StubBackend(), vla_candidates=16, action_chunk_size=1,
    ))
    monkeypatch.setattr(proxy, "_select_grouped_vla_response", select)
    session = StubSession([_observation(i) for i in range(10)])
    asyncio.run(proxy._handle_session(session))
    assert len(session.sent[0]) == 10
    assert len(proxy.config.backend.connection.calls[0]) == 160


def test_gemini_request_has_no_output_token_cap(monkeypatch):
    captured = {}
    vlm = GoogleVLM(api_key="test")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}]}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(vlm.session, "post", post)
    assert vlm.generate("system", "text", np.zeros((8, 8, 3), dtype=np.uint8)) == '{"ok":true}'
    assert "/models/gemini-3.8-flash:generateContent" in captured["url"]
    assert captured["json"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "MEDIUM"
    assert "maxOutputTokens" not in captured["json"]["generationConfig"]
    assert "max_tokens" not in captured["json"]
