import numpy as np
import pytest

from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy
from vlm_orchestrator.vlm.api import (
    DecompositionResult,
    GoogleVLM,
    ProgressResult,
    infer_atomic_action,
)
from vlm_orchestrator.vlm.task_descriptions import TASK_REFERENCES, find_task_reference


class StubVLM:
    def __init__(self):
        self.progress = [
            ProgressResult("stay"),
            ProgressResult("advance", grasp_secure=True, object_lifted=True),
            ProgressResult("stay"),
        ]
        self.progress_calls = []
        self.memory_calls = []
        self.replan_memories = []
        self.replan_contexts = []
        self.progress_contexts = []
        self.decompose_calls = []
        self.executed_trajectory_calls = []

    def decompose(
        self, instruction, image, extra_images=None, *, task_description="",
        full_score_condition="",
    ):
        self.decompose_calls.append(
            (instruction, task_description, full_score_condition)
        )
        return DecompositionResult(
            subgoals=[
                "Pick the red block with the left arm at [40, 80].",
                "Place the red block with the left arm at [120, 160].",
            ],
            ordered=True,
            coordinates=[[[40, 80]], [[120, 160]]],
            arms=["left", "left"],
            atomic_actions=["pick", "place"],
            adapter_ids=["", ""],
        )

    def assess_progress(
        self, instruction, current_subgoal, remaining_subgoals,
        last_advanced_subgoal, image,
        extra_images=None, *, before_images=None, memory="",
        last_executed_ee_trajectory=None,
        task_description="", full_score_condition="",
    ):
        self.memory_calls.append(memory)
        self.executed_trajectory_calls.append(last_executed_ee_trajectory)
        self.progress_calls.append(
            (current_subgoal, list(remaining_subgoals), last_advanced_subgoal)
        )
        self.progress_contexts.append((task_description, full_score_condition))
        result = self.progress.pop(0)
        return result if isinstance(result, ProgressResult) else ProgressResult(result)

    def replan(
        self, instruction, current_subgoal, remaining_subgoals, reason, image,
        extra_images=None, *, before_images=None, memory="",
        task_description="", full_score_condition="",
        last_executed_ee_trajectory=None,
    ):
        self.replan_memories.append(memory)
        self.executed_trajectory_calls.append(last_executed_ee_trajectory)
        self.replan_contexts.append((task_description, full_score_condition))
        return DecompositionResult(
            subgoals=["Pick the corrected target with the right arm at [90, 110]."],
            ordered=True,
            coordinates=[[[90, 110]]],
            arms=["right"],
            atomic_actions=["pick"],
            adapter_ids=[""],
        )


def _observation():
    return {
        "prompt": "Move the red block.",
        "observation/exterior_image_1_left": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def test_official_robodojo_catalog_covers_all_simulation_tasks():
    assert len(TASK_REFERENCES) == 43
    assert len({reference.slug for reference in TASK_REFERENCES}) == 43
    assert "key and a keyhole" in find_task_reference("insert_key").description
    scored = [reference for reference in TASK_REFERENCES if reference.full_score_condition]
    assert len(scored) == 42
    assert find_task_reference("dlc").full_score_condition == ""
    press_score = find_task_reference("press_by_number").full_score_condition
    assert press_score.count("blue confirm button is pressed") == 2
    assert "robot returns to origin" in press_score


def test_subgoal_strategy_only_stays_or_advances_one_subgoal():
    vlm = StubVLM()
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm), SubgoalConfig(progress_interval=1)
    )
    state = SessionState()

    first, state = strategy.process(_observation(), state)
    assert state.current_subgoal_idx == 0
    assert first["prompt"].startswith("Pick the red block")
    assert first["coordinates"] == [40, 80]

    state.episode_step = 1
    stayed, state = strategy.process(first, state)
    assert state.current_subgoal_idx == 0
    assert stayed["prompt"].startswith("Pick the red block")

    state.episode_step = 2
    advanced, state = strategy.process(stayed, state)
    assert state.current_subgoal_idx == 1
    assert advanced["prompt"].startswith("Place the red block")
    assert advanced["coordinates"] == [120, 160]
    assert state.flush_actions is False

    state.episode_step = 3
    final, state = strategy.process(advanced, state)
    assert state.current_subgoal_idx == 1
    assert final["prompt"].startswith("Place the red block")
    assert vlm.progress_calls == [
        (
            "Pick the red block with the left arm at [40, 80].",
            ["Place the red block with the left arm at [120, 160]."],
            None,
        ),
        (
            "Pick the red block with the left arm at [40, 80].",
            ["Place the red block with the left arm at [120, 160]."],
            None,
        ),
        (
            "Place the red block with the left arm at [120, 160].",
            [],
            "Pick the red block with the left arm at [40, 80].",
        ),
    ]


def test_progress_transition_requires_a_fresh_chunk_before_next_advance():
    strategy = SubgoalStrategy(StrategyContext(vlm=StubVLM()))
    state = SessionState(
        subgoals=["first", "second"],
        subgoal_atomic_actions=["move", "move"],
        subgoal_coordinates=[[], []],
        progress_transition_pending=True,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    _, changed = strategy._apply_progress_result(
        {}, state, {"decision": "advance", "reason": "tail motion"},
        image, [], "select_trajectory_group",
    )
    assert changed is False
    assert state.current_subgoal_idx == 0
    assert state.progress_transition_pending is False
    assert state.log_entries[-1]["transition_guard_blocked"] is True
    assert state.log_entries[-1]["decision"] == "stay"

    _, changed = strategy._apply_progress_result(
        {}, state, {"decision": "advance", "reason": "fresh completion"},
        image, [], "select_trajectory_group",
    )
    assert changed is True
    assert state.current_subgoal_idx == 1
    assert state.progress_transition_pending is True


def test_final_subgoal_sets_task_completed():
    strategy = SubgoalStrategy(StrategyContext(vlm=StubVLM()))
    state = SessionState(
        subgoals=["only"],
        subgoal_atomic_actions=["move"],
        subgoal_coordinates=[[]],
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    _, changed = strategy._apply_progress_result(
        {}, state, {"decision": "advance", "reason": "complete"},
        image, [], "select_trajectory_group",
    )
    assert changed is True
    assert state.task_completed is True
    assert state.log_entries[-1]["type"] == "task_completed"


def test_completed_strategy_does_not_replan_or_check_progress_again():
    vlm = StubVLM()
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm), SubgoalConfig(progress_interval=1)
    )
    state = SessionState(
        original_instruction="Move the red block.",
        rewritten_instruction="Move the red block.",
        subgoals=["already done"],
        subgoal_atomic_actions=["press"],
        subgoal_coordinates=[[]],
        task_completed=True,
        pending_replan_reason="stale selector request",
        episode_step=10,
        step_at_last_progress_check=0,
    )

    strategy.process(_observation(), state)

    assert state.task_completed is True
    assert state.pending_replan_reason == "stale selector request"
    assert vlm.progress_calls == []
    assert vlm.replan_memories == []


def test_subgoal_strategy_replans_subgoals_and_coordinates_on_misjudgment():
    vlm = StubVLM()
    vlm.progress = [ProgressResult("replan")]
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm), SubgoalConfig(progress_interval=1)
    )
    state = SessionState()

    first, state = strategy.process(_observation(), state)
    previous_revision = state.target_revision
    state.episode_step = 1
    corrected, state = strategy.process(first, state)

    assert state.subgoals == [
        "Pick the corrected target with the right arm at [90, 110]."
    ]
    assert state.subgoal_coordinates == [[[90, 110]]]
    assert state.current_subgoal_idx == 0
    assert state.target_revision == previous_revision + 1
    assert state.flush_actions is False
    assert corrected["prompt"].startswith("Pick the corrected target")
    assert corrected["coordinates"] == [90, 110]
    assert "arm" not in corrected


def test_unchanged_replan_preserves_previous_trajectory_and_subgoal_position():
    vlm = StubVLM()
    strategy = SubgoalStrategy(StrategyContext(vlm=vlm))
    observation, state = strategy.process(_observation(), SessionState())
    state.current_subgoal_idx = 1
    state.last_advanced_subgoal = state.subgoals[0]
    state.target_revision = 1
    state.action_cache = np.zeros((30, 14))
    state.action_cache_offset = 10
    state.action_cache_target_revision = 1
    state.episode_step = 20
    state.pending_replan_reason = "check the selected trajectory"
    vlm.replan = lambda *args, **kwargs: DecompositionResult(
        subgoals=[state.subgoals[1]],
        ordered=state.subgoals_ordered,
        coordinates=[state.subgoal_coordinates[1]],
        arms=[state.subgoal_arms[1]],
        atomic_actions=[state.subgoal_atomic_actions[1]],
        adapter_ids=[state.subgoal_adapters[1]],
    )

    updated, state = strategy.process(observation, state)

    assert state.current_subgoal_idx == 1
    assert state.last_advanced_subgoal == state.subgoals[0]
    assert state.target_revision == state.action_cache_target_revision == 1
    assert state.action_cache_offset == 10
    assert state.action_cache.shape == (30, 14)
    assert state.flush_actions is False
    assert state.pending_replan_reason is None
    assert state.step_at_last_progress_check == 20
    assert updated["prompt"] == state.subgoals[1]
    assert updated["coordinates"] == [120, 160]
    assert state.log_entries[-1]["type"] == "subgoal_replan_unchanged"


def test_replan_changes_only_future_subgoals_without_flushing_current_trajectory():
    vlm = StubVLM()
    strategy = SubgoalStrategy(StrategyContext(vlm=vlm))
    observation, state = strategy.process(_observation(), SessionState())
    state.action_cache = np.zeros((30, 14))
    state.action_cache_offset = 10
    state.action_cache_target_revision = 0
    state.pending_replan_reason = "update the later placement"
    vlm.replan = lambda *args, **kwargs: DecompositionResult(
        subgoals=[state.subgoals[0], "Place the red block with the right arm at [90, 110]."],
        ordered=True,
        coordinates=[state.subgoal_coordinates[0], [[90, 110]]],
        arms=["left", "right"],
        atomic_actions=["pick", "place"],
        adapter_ids=["", ""],
    )

    updated, state = strategy.process(observation, state)

    assert state.subgoals[1] == "Place the red block with the right arm at [90, 110]."
    assert updated["coordinates"] == [40, 80]
    assert state.target_revision == state.action_cache_target_revision == 0
    assert state.action_cache_offset == 10
    assert state.flush_actions is False


def test_replan_with_new_current_coordinates_retains_previous_trajectory():
    vlm = StubVLM()
    strategy = SubgoalStrategy(StrategyContext(vlm=vlm))
    observation, state = strategy.process(_observation(), SessionState())
    state.action_cache = np.zeros((30, 14))
    state.action_cache_offset = 10
    state.action_cache_target_revision = 0
    state.pending_replan_reason = "target moved"
    vlm.replan = lambda *args, **kwargs: DecompositionResult(
        subgoals=list(state.subgoals),
        ordered=True,
        coordinates=[[[90, 110]], state.subgoal_coordinates[1]],
        arms=list(state.subgoal_arms),
        atomic_actions=list(state.subgoal_atomic_actions),
        adapter_ids=list(state.subgoal_adapters),
    )

    updated, state = strategy.process(observation, state)

    assert updated["coordinates"] == [90, 110]
    assert state.target_revision == 1
    assert state.action_cache_target_revision == 0
    assert state.flush_actions is False


def test_check_memory_rolls_forward_to_check_and_replan_and_resets_on_new_episode():
    vlm = StubVLM()
    vlm.progress = [
        ProgressResult("stay", memory="The first grasp missed the red block."),
        ProgressResult(
            "replan", "target needs correction",
            memory="The first grasp missed; the current target point is also wrong.",
        ),
    ]
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm), SubgoalConfig(progress_interval=1)
    )
    observation, state = strategy.process(_observation(), SessionState())

    state.episode_step = 1
    state.last_executed_ee_trajectory = {"executed_steps": 10, "steps": []}
    observation, state = strategy.process(observation, state)
    assert state.check_memory == "The first grasp missed the red block."

    state.episode_step = 2
    observation, state = strategy.process(observation, state)
    assert vlm.memory_calls == ["", "The first grasp missed the red block."]
    assert vlm.replan_memories == [
        "The first grasp missed; the current target point is also wrong."
    ]
    assert state.check_memory == vlm.replan_memories[0]
    assert vlm.executed_trajectory_calls == [
        {"executed_steps": 10, "steps": []},
        {"executed_steps": 10, "steps": []},
        {"executed_steps": 10, "steps": []},
    ]

    new_episode = _observation()
    new_episode["prompt"] = "Move a different object."
    _, state = strategy.process(new_episode, state)
    assert state.check_memory == ""
    assert state.last_executed_ee_trajectory is None


def test_replan_accepts_missing_coordinates_and_clears_previous_target():
    vlm = StubVLM()
    vlm.progress = [ProgressResult("replan"), ProgressResult("advance")]
    vlm.replan = lambda *args, **kwargs: DecompositionResult(
        subgoals=["Place the block with the left arm.", "Lift the block with the right arm."],
        ordered=True,
        coordinates=[[[90, 110]], []],
        arms=["left", "right"],
        atomic_actions=["place", "lift_up"],
    )
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm),
        SubgoalConfig(progress_interval=1),
    )
    observation, state = strategy.process(_observation(), SessionState())
    state.episode_step = 1
    observation, state = strategy.process(observation, state)

    assert state.subgoal_coordinates == [[[90, 110]], []]
    assert state.target_revision == 1
    assert state.flush_actions is False
    assert observation["coordinates"] == [90, 110]

    state.episode_step = 2
    observation, state = strategy.process(observation, state)
    assert state.current_subgoal_idx == 1
    assert "arm" not in observation
    assert "coordinates" not in observation


@pytest.mark.skip(reason="pick-specific completion gate removed; VLM decides generically")
def test_pick_does_not_advance_without_secure_grasp_and_lift():
    vlm = StubVLM()
    vlm.progress = [
        ProgressResult("advance", "gripper is near the object", grasp_secure=False),
        ProgressResult("advance", "object is held", grasp_secure=True, object_lifted=False),
        ProgressResult(
            "advance",
            "object is held and lifted",
            grasp_secure=True,
            object_lifted=True,
        ),
    ]
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm), SubgoalConfig(progress_interval=1)
    )
    state = SessionState()
    observation, state = strategy.process(_observation(), state)

    for step in (1, 2):
        state.episode_step = step
        observation, state = strategy.process(observation, state)
        assert state.current_subgoal_idx == 0
        assert observation["prompt"].startswith("Pick the red block")
        assert state.log_entries[-1]["requested_decision"] == "advance"
        assert state.log_entries[-1]["decision"] == "stay"

    state.episode_step = 3
    observation, state = strategy.process(observation, state)
    assert state.current_subgoal_idx == 1
    assert observation["prompt"].startswith("Place the red block")


@pytest.mark.skip(reason="pick-specific completion gate removed; VLM decides generically")
def test_pick_advances_after_secure_grasp_and_consecutive_ee_lift(monkeypatch):
    vlm = StubVLM()
    vlm.progress = [
        ProgressResult("stay", "approaching", grasp_secure=False, object_lifted=False),
        ProgressResult("stay", "grasped but occluded", grasp_secure=True, object_lifted=False),
        ProgressResult("stay", "still occluded", grasp_secure=True, object_lifted=False),
    ]
    heights = iter([0.80, 0.84, 0.85])
    monkeypatch.setattr(
        "vlm_orchestrator.strategies.subgoal.fk_trajectory",
        lambda chunk: {"left": np.array([[0.0, 0.0, next(heights)]]),
                       "right": np.array([[0.0, 0.0, 0.0]])},
    )
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm),
        SubgoalConfig(
            progress_interval=1, pick_lift_height_m=0.03,
            pick_lift_confirmations=2,
        ),
    )
    obs = _observation()
    obs["observation/joint_position"] = np.zeros(12)
    observation, state = strategy.process(obs, SessionState())

    state.episode_step = 1
    observation, state = strategy.process(observation, state)
    assert state.pick_pregrasp_ee_z == 0.80
    assert state.current_subgoal_idx == 0

    state.episode_step = 2
    observation, state = strategy.process(observation, state)
    assert state.pick_lift_evidence_count == 1
    assert state.current_subgoal_idx == 0

    state.episode_step = 3
    observation, state = strategy.process(observation, state)
    progress = next(entry for entry in reversed(state.log_entries) if entry.get("operation") == "assess_progress")
    assert progress["requested_decision"] == "stay"
    assert progress["decision"] == "advance"
    assert progress["vlm_object_lifted"] is False
    assert progress["object_lifted"] is True
    assert progress["ee_lift_confirmed"] is True
    assert progress["ee_lift_m"] == pytest.approx(0.05)
    assert state.current_subgoal_idx == 1
    assert state.pick_lift_evidence_count == 0


@pytest.mark.skip(reason="pick-specific completion gate removed; VLM decides generically")
def test_first_secure_grasp_height_updates_descent_baseline(monkeypatch):
    vlm = StubVLM()
    vlm.progress = [
        ProgressResult("stay", grasp_secure=False),
        ProgressResult("stay", grasp_secure=True),
        ProgressResult("stay", grasp_secure=True),
        ProgressResult("stay", grasp_secure=True),
    ]
    heights = iter([0.881, 0.816, 0.892, 0.853])
    monkeypatch.setattr(
        "vlm_orchestrator.strategies.subgoal.fk_trajectory",
        lambda chunk: {"left": np.array([[0.0, 0.0, next(heights)]]),
                       "right": np.array([[0.0, 0.0, 0.0]])},
    )
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm),
        SubgoalConfig(
            progress_interval=1, pick_lift_height_m=0.03,
            pick_lift_confirmations=2,
        ),
    )
    obs = _observation()
    obs["observation/joint_position"] = np.zeros(12)
    observation, state = strategy.process(obs, SessionState())
    for step in (1, 2):
        state.episode_step = step
        observation, state = strategy.process(observation, state)
    assert state.pick_pregrasp_ee_z == pytest.approx(0.816)
    assert state.current_subgoal_idx == 0
    state.episode_step = 3
    observation, state = strategy.process(observation, state)
    assert state.pick_lift_evidence_count == 1
    state.episode_step = 4
    observation, state = strategy.process(observation, state)
    progress = next(entry for entry in reversed(state.log_entries) if entry.get("operation") == "assess_progress")
    assert progress["ee_lift_m"] == pytest.approx(0.037)
    assert progress["ee_lift_confirmed"] is True
    assert state.current_subgoal_idx == 1


class StubGoogleVLM(GoogleVLM):
    def __init__(self, responses):
        self.responses = list(responses)
        self.schemas = []
        self.requests = []

    def generate(
        self, system, text, image, extra_images=None, response_schema=None
    ):
        self.schemas.append(response_schema)
        self.requests.append((text, image, extra_images))
        return self.responses.pop(0)


def test_decompose_and_replan_describe_arm_as_non_binding_hint():
    from vlm_orchestrator.vlm.api import (
        DECOMPOSE_PROMPT_SPEC,
        DECOMPOSE_RESPONSE_SCHEMA,
        REPLAN_PROMPT_SPEC,
    )

    for prompt in (DECOMPOSE_PROMPT_SPEC, REPLAN_PROMPT_SPEC):
        prompt = " ".join(prompt.split())
        assert "`arm` is required for every subgoal" in prompt
        assert "non-binding planning hint" in prompt
        assert "selector may use the other arm" in prompt
        assert "prefer the left arm" not in prompt
    required = DECOMPOSE_RESPONSE_SCHEMA["properties"]["subgoals"]["items"]["required"]
    assert "arm" in required


def test_before_views_precede_current_views_in_progress_and_replan():
    images = [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(6)]
    vlm = StubGoogleVLM([
        '{"decision":"replan","reason":"target changed","grasp_secure":false,'
        '"object_lifted":false,"memory":"The old target is no longer valid."}',
        '{"subgoals":[{"text":"Pick the cube.","arm":"left",'
        '"atomic_action":"pick","coordinates":[[80,120]]}]}',
    ])
    vlm.assess_progress(
        "Move cube", "Pick cube", [], None, images[3], images[4:],
        before_images=images[:3], memory="The cube was previously on the left.",
    )
    vlm.replan(
        "Move cube", "Pick cube", [], "target changed", images[3], images[4:],
        before_images=images[:3], memory="The old target is no longer valid.",
        task_description="Move the task cube to its target.",
        full_score_condition="The cube is on the target and the robot returns to origin.",
        last_executed_ee_trajectory={"executed_steps": 10, "steps": []},
    )
    for text, image, extras in vlm.requests:
        assert "first 3 are BEFORE" in text
        assert [int(value[0, 0, 0]) for value in [image, *extras]] == list(range(6))
    assert "Previous check memory: The cube was previously on the left." in vlm.requests[0][0]
    assert "Make decision_reason and reason quantitative whenever possible" in vlm.requests[0][0]
    assert "no numeric evidence available" in vlm.requests[0][0]
    assert "Any advance decision MUST be justified only by the already executed" in vlm.requests[0][0]
    assert "Do not use any candidate or predicted trajectory as completion evidence" in vlm.requests[0][0]
    for action in (
        "observe", "pick", "place", "insert", "handover", "press", "turn",
        "throw", "hang", "fold", "push", "close", "strike", "pour", "sweep",
        "reorient", "lift_up",
    ):
        assert f'"{action}"' in vlm.requests[0][0]
    assert "Accumulated check memory: The old target is no longer valid." in vlm.requests[1][0]
    assert "Official RoboDojo task description" in vlm.requests[1][0]
    assert "Official RoboDojo full-score condition" in vlm.requests[1][0]
    assert "Recent executed EE trajectory (up to 20 steps; each action chunk is normally 10 steps;" in vlm.requests[1][0]


def test_strategy_caches_before_views_as_independent_copies():
    vlm = StubVLM()
    captured = []

    def progress(
        *args, before_images=None, memory="", last_executed_ee_trajectory=None,
        **kwargs,
    ):
        captured.append([int(value[0, 0, 0]) for value in before_images])
        return ProgressResult("stay")

    vlm.assess_progress = progress
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm),
        SubgoalConfig(progress_interval=1),
    )
    obs = _observation()
    obs["observation/wrist_image_left"] = np.full((8, 8, 3), 1, dtype=np.uint8)
    obs["observation/wrist_image_right"] = np.full((8, 8, 3), 2, dtype=np.uint8)
    _, state = strategy.process(obs, SessionState())
    for key in ("observation/exterior_image_1_left", "observation/wrist_image_left", "observation/wrist_image_right"):
        obs[key][:] += 3
    state.episode_step = 1
    strategy.process(obs, state)
    state.episode_step = 2
    strategy.process(obs, state)
    assert captured == [[0, 1, 2], [3, 4, 5]]


def test_google_vlm_contract_decomposes_assesses_and_replans():
    vlm = StubGoogleVLM(
        [
            '{"subgoals":[{"text":"Pick the cube.","arm":"right",'
            '"atomic_action":"observe",'
            '"coordinates":[[25,75]]}]}',
            '{"decision":"replan","reason":"target point is wrong",'
            '"grasp_secure":false,"object_lifted":false,'
            '"memory":"The original target point was incorrect."}',
            '{"subgoals":[{"text":"Pick the corrected cube.","arm":"left",'
            '"atomic_action":"pick",'
            '"coordinates":[[100,120]]}]}',
        ]
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    plan = vlm.decompose("Pick the cube.", image)
    progress = vlm.assess_progress(
        "Pick the cube.", plan.subgoals[0], [], None, image
    )
    corrected = vlm.replan(
        "Pick the cube.",
        plan.subgoals[0],
        [],
        progress.reason,
        image,
    )

    assert plan.coordinates == [[[25, 75]]]
    assert plan.arms == ["right"]
    assert plan.atomic_actions == ["observe"]
    assert progress.decision == "replan"
    assert progress.reason == "target point is wrong"
    assert progress.memory == "The original target point was incorrect."
    assert corrected.coordinates == [[[100, 120]]]
    assert corrected.arms == ["left"]
    assert corrected.atomic_actions == ["pick"]
    assert set(vlm.schemas[0]["properties"]) == {"subgoals"}
    assert "atomic_action" in vlm.schemas[0]["properties"]["subgoals"]["items"]["required"]
    assert "coordinates" not in vlm.schemas[0]["properties"]["subgoals"]["items"]["required"]
    assert set(vlm.schemas[1]["properties"]) == {
        "decision", "reason", "decision_reason", "memory", "before_to_now_summary",
    }
    assert set(vlm.schemas[2]["properties"]) == {"subgoals"}


def test_google_vlm_normalizes_1000_scale_coordinates():
    vlm = StubGoogleVLM([
        '{"subgoals":['
        '{"text":"Press the left button at [380, 555].","atomic_action":"press",'
        '"coordinates":[[380,555]]},'
        '{"text":"Press the middle button at [501, 555].","atomic_action":"press",'
        '"coordinates":[[501,555]]},'
        '{"text":"Press the blue button at [618, 555].","atomic_action":"press",'
        '"coordinates":[[618,555]]}'
        ']}'
    ])

    plan = vlm.decompose("Press the buttons.", np.zeros((480, 640, 3), dtype=np.uint8))

    assert plan.coordinates == [[[97, 142]], [[128, 142]], [[158, 142]]]
    assert plan.subgoals == [
        "Press the left button at [97, 142].",
        "Press the middle button at [128, 142].",
        "Press the blue button at [158, 142].",
    ]


def test_google_vlm_decompose_includes_official_task_description():
    vlm = StubGoogleVLM([
        '{"subgoals":[{"text":"Press the red button.","arm":"left",'
        '"atomic_action":"press"}]}',
    ])

    vlm.decompose(
        "Press the requested button.",
        np.zeros((8, 8, 3), dtype=np.uint8),
        task_description="Press each red button, then press blue to confirm.",
        full_score_condition=(
            "Press blue after the first red button, press the second red button, "
            "then press blue again."
        ),
    )

    prompt = vlm.requests[0][0]
    assert 'Task instruction: "Press the requested button."' in prompt
    assert "Official RoboDojo task description" in prompt
    assert "then press blue to confirm" in prompt
    assert "Official RoboDojo full-score condition" in prompt
    assert "then press blue again" in prompt


def test_strategy_uses_task_name_for_decomposition_description_only():
    vlm = StubVLM()
    strategy = SubgoalStrategy(StrategyContext(vlm=vlm))
    observation = _observation()
    observation["prompt"] = "Do the requested button task."
    observation["__task_name"] = "press_by_number"

    rewritten, state = strategy.process(observation, SessionState())

    instruction, description, full_score_condition = vlm.decompose_calls[0]
    assert instruction == "Do the requested button task."
    assert "two number cards" in description
    assert "blue confirmation button" in description
    assert full_score_condition.count("blue confirm button is pressed") == 2
    assert "robot returns to origin" in full_score_condition
    assert state.original_instruction == "Do the requested button task."
    assert rewritten["prompt"].startswith("Pick the red block")


def test_strategy_passes_official_task_context_to_replan():
    vlm = StubVLM()
    strategy = SubgoalStrategy(StrategyContext(vlm=vlm))
    observation = _observation()
    observation["prompt"] = "Do the requested button task."
    observation["__task_name"] = "press_by_number"
    observation, state = strategy.process(observation, SessionState())
    state.pending_replan_reason = "refresh the remaining button sequence"

    strategy.process(observation, state)

    description, full_score_condition = vlm.replan_contexts[0]
    assert "two number cards" in description
    assert full_score_condition.count("blue confirm button is pressed") == 2


def test_strategy_passes_official_task_context_to_progress_check():
    vlm = StubVLM()
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm), SubgoalConfig(progress_interval=1)
    )
    observation = _observation()
    observation["prompt"] = "Do the requested button task."
    observation["__task_name"] = "press_by_number"
    observation, state = strategy.process(observation, SessionState())
    state.episode_step = 1

    strategy.process(observation, state)

    description, full_score_condition = vlm.progress_contexts[0]
    assert "two number cards" in description
    assert full_score_condition.count("blue confirm button is pressed") == 2


def test_google_vlm_accepts_subgoal_without_coordinates():
    vlm = StubGoogleVLM([
        '{"subgoals":[{"text":"Raise the held object.","arm":"left",'
        '"atomic_action":"lift_up"}]}',
    ])

    plan = vlm.decompose(
        "Raise the held object.", np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert plan.atomic_actions == ["lift_up"]
    assert plan.coordinates == [[]]
    assert len(vlm.requests) == 1


def test_subgoal_strategy_preserves_observe_atomic_action():
    vlm = StubVLM()
    vlm.decompose = lambda instruction, image, extra_images=None: DecompositionResult(
        subgoals=["Observe the red block with the left arm at [40, 80]."],
        ordered=True,
        coordinates=[[[40, 80]]],
        arms=["left"],
        atomic_actions=["observe"],
        adapter_ids=[""],
    )
    strategy = SubgoalStrategy(
        StrategyContext(vlm=vlm), SubgoalConfig(progress_interval=1)
    )

    observation, state = strategy.process(_observation(), SessionState())

    assert state.subgoal_atomic_actions == ["observe"]
    assert observation["atomic_action"] == "observe"
    assert observation["coordinates"] == [40, 80]

    state.episode_step = 1
    strategy.process(observation, state)
    assert len(vlm.progress_calls) == 1


def test_observe_can_be_inferred_from_explicit_observation_verbs():
    assert infer_atomic_action("Observe the target area.") == "observe"
    assert infer_atomic_action("Watch the indicator.") == "observe"
    assert infer_atomic_action("Inspect the slot.") == "observe"
