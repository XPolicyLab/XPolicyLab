"""Evaluator transport for original standard/random public number generators.

No task ordering, scoring or alternate controller lives here. The unchanged
external generator emits each already-sampled action; official take_action ACKs
it once. Only source-domain failures continue finite held targets to terminal.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from harness.runtime.public_policy_recording import PublicEvidence

ROOT = Path(__file__).resolve().parents[2]


def prepare_runtime():
    from curobo.batch_motion_planner import BatchMotionPlanner
    from curobo.motion_planner import MotionPlanner
    MotionPlanner.warmup = lambda self, *arguments, **keywords: True
    BatchMotionPlanner.warmup = lambda self, *arguments, **keywords: True
    from env.planner_manager.curobo_planner import CuroboPlanner
    CuroboPlanner._prewarm_alternate_modes = lambda self: None
    from env.robot_manager.robot_manager import RobotManager

    def setup_planner(manager, robot):
        planner = CuroboPlanner(robot_origin_pose=[0, 0, 0, 1, 0, 0, 0], active_joints_name=robot.arm_joints_name, all_joints=robot.arm_joints_name, dt=manager.dt, yml_path=robot.curobo_yml_path, table_height=-0.025)
        manager.planner[robot.robot_name] = manager.ik_solver[robot.robot_name] = planner
    RobotManager._setup_planner = setup_planner
    original_build = CuroboPlanner._build_robot_and_scene_cfg

    def reserve(planner, *arguments, **keywords):
        (robot, scene) = original_build(planner, *arguments, **keywords)
        for index in range(40):
            scene['cuboid']['reserved_rgb_' + str(index)] = dict(dims=[0.001, 0.001, 0.001], pose=[0, 0, -10 - index, 1, 0, 0, 0])
        return (robot, scene)
    CuroboPlanner._build_robot_and_scene_cfg = reserve


def public_observation(raw, count, protocol, summary=''):
    state = {key: raw['state'][key] for key in protocol.STATE_KEYS}
    vision = {camera: raw['vision'][camera]['colors'][..., :3] for camera in protocol.CAMERAS}
    return protocol.Observation.create(state, vision, count, summary)


def source_domain_refusal(error, task):
    visited, current = set(), error
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = (type(current).__name__ + ': ' + str(current)).lower()
        if any(marker in message for marker in ('outofmemory', 'out of memory', 'cuda error',
                                                'device-side assert', 'illegal memory access')):
            return False
        current = current.__cause__ or current.__context__
    trace = error.__traceback__
    if trace is None:
        return False
    while trace.tb_next is not None:
        trace = trace.tb_next
    source = Path(trace.tb_frame.f_code.co_filename).resolve()
    root = ROOT
    return (source.is_relative_to(Path(task).resolve())
            or source in {root / member for member in ('tasks/arrange_largest_number_random/skills/action_commands/implementation.py', 'tasks/arrange_largest_number_random/skills/context/implementation.py', 'tasks/arrange_largest_number_random/skills/execution_state/implementation.py', 'tasks/arrange_largest_number_random/skills/identity_recovery/implementation.py', 'tasks/arrange_largest_number_random/skills/placement_state/implementation.py', 'tasks/arrange_largest_number_random/skills/scene_recovery/implementation.py', 'tasks/arrange_largest_number_random/skills/source_recovery/implementation.py', 'tasks/arrange_largest_number_random/skills/target_selection/implementation.py', 'tasks/arrange_largest_number_random/skills/transfer/implementation.py', 'skills/shared/perception/material_glyph_outline/implementation.py', 'skills/shared/perception/glyph_template_bank/implementation.py', 'skills/shared/perception/glyph_outline_search/implementation.py', 'skills/shared/perception/glyph_scene_observation/implementation.py', 'skills/shared/motion/feedback_joint_transport/implementation.py', 'skills/shared/geometry/gripper_target_frames/implementation.py', 'skills/shared/geometry/x5_public_frame_model/implementation.py', 'skills/shared/geometry/head_wrist_camera_calibration/implementation.py', 'skills/shared/support/verified_geometry_assets/implementation.py', 'skills/shared/observability/public_action_observation/implementation.py', 'skills/shared/planning/public_joint_path_planning/implementation.py')}
            or source == root / 'harness/runtime/declarative_generator_workflow.py')


def drive(env, recorder, exchange, actor, protocol, task):
    observation = public_observation(recorder.public.last_observation,
                                     int(env.take_action_cnt[0]), protocol)
    refusal = None
    held = {key: np.asarray(observation.state[key]).tolist()
            for key in ('left_arm_joint_state', 'left_ee_joint_state',
                        'right_arm_joint_state', 'right_ee_joint_state')}
    exchange.phase = lambda: actor.stage if refusal is None else 'original_refusal_final_hold'
    try:
        actor.update(observation)
        while not env.end_flag[0]:
            command = protocol.validate_bound_action(actor.next_action(), observation)
            before = observation.physical_step
            raw, ended = exchange(command)
            count = int(env.take_action_cnt[0])
            if count != before + 1:
                raise RuntimeError('Original generator ACK must advance exactly one action')
            held = deepcopy(command)
            updated = public_observation(raw, count, protocol, actor.summary())
            actor.update(updated)
            observation = updated
    except Exception as error:
        if not source_domain_refusal(error, task):
            raise
        refusal = {'type': type(error).__name__, 'message': str(error),
                   'action_count': int(env.take_action_cnt[0])}
        while not env.end_flag[0]:
            exchange(held)
    finally:
        actor._integrated_source_refusal = refusal
        actor.close()
    return {'original_source_refusal': refusal, 'score_authority': 'official_run_eval_finalcheck'}


def diagnostics(actor):
    return {'summary': actor.summary(),
            'original_source_refusal': getattr(actor, '_integrated_source_refusal', None)}
