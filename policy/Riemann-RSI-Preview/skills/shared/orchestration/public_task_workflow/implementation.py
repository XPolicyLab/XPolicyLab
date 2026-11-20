"""Existing public task factories behind the real official policy protocol."""
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import os
import traceback

import numpy as np

from skills.shared.motion.public_observation_command_clock.implementation import (
    PublicCommandClock, PublicActionBudgetReached, color_frame,
)
from skills.shared.orchestration.public_mature_workflow.implementation import save, configure_planners
from skills.shared.orchestration.public_rule_policy_interface.implementation import normalized_action

ROOT = Path(__file__).resolve().parents[4]
FAMILIES = frozenset(('public_observation_policy', 'public_command_workflow',
                      'public_stack_workflow', 'public_toast_workflow'))
_prepared = set()


def prepare(mode, source):
    """Only policy-local numeric setup, once per server; never evaluator hooks."""
    if mode in _prepared:
        return
    configure_planners(False)
    if mode.startswith('arrange_largest_number'):
        from env.planner_manager.curobo_planner import CuroboPlanner
        original = CuroboPlanner._build_robot_and_scene_cfg
        def reserve(planner, *args, **kwargs):
            robot, scene = original(planner, *args, **kwargs)
            for index in range(40):
                scene['cuboid']['reserved_rgb_' + str(index)] = dict(
                    dims=[.001, .001, .001], pose=[0, 0, -10-index, 1, 0, 0, 0])
            return robot, scene
        CuroboPlanner._build_robot_and_scene_cfg = reserve
    elif mode == 'sort_nesting_dolls_by_size':
        source.assets.verify_public_resources(ROOT / 'tasks' / mode)
        from env.planner_manager.curobo_planner import CuroboPlanner
        source.reserve_obstacle_slots(CuroboPlanner)
    elif source is not None and hasattr(source, 'prepare_runtime'):
        source.prepare_runtime()
    _prepared.add(mode)


class PublicSession:
    """A policy-owned observation history and ACK counter, not an EvalEnv."""
    def __init__(self, initial, rpc, root, horizon, evidence, *, robots=None, task_name=''):
        self.output_root = Path(root)
        self.public = evidence
        self._rpc, self._robots = rpc, robots
        self.environment = SimpleNamespace(dt=.004, step_lim=int(horizon),
            task_name=str(task_name), num_envs=1,
            obs_manager=SimpleNamespace(collect_interval=10, collect_freq=25),
            end_flag=np.array([False]), take_action_cnt=np.array([0]))
        self.phase = lambda: 'public_workflow'
        self.normalized_actions, self.action_sim_steps, self.action_status = [], [], []
        self.phases = ['reset']
        self._append(initial)

    def _append(self, frame):
        self.current_public_observation = color_frame(frame)
        if self._robots is None:
            self.public.append(frame, None)
            # PublicEvidence intentionally drops unrelated observation metadata.
            # Only this explicit public field is retained for language actors.
            self.public.instruction = frame.get('instruction', '')
        else:
            self.public.append(self.current_public_observation, self._robots)

    def __call__(self, action):
        if self.environment.end_flag[0]:
            raise PublicActionBudgetReached('Own acknowledged-command budget reached')
        target = normalized_action(action)
        frame = self._rpc(deepcopy(target))
        self.environment.take_action_cnt[0] += 1
        count = int(self.environment.take_action_cnt[0])
        phase = str(self.phase())
        self.normalized_actions.append(target)
        self.action_sim_steps.append(count * 10)
        self.action_status.append({'phase': phase, 'own_action_count': count})
        self.phases.append(phase)
        self._append(frame)
        self.environment.end_flag[0] = count >= self.environment.step_lim
        return self.public_frame(), bool(self.environment.end_flag[0])

    def public_frame(self):
        frame = self.current_public_observation
        return {'state': deepcopy(frame['state']),
                'vision': {name: {'colors': view['color'].copy()} for name, view in frame['vision'].items()}}


class _ClockHistory:
    """Connect unchanged dense sampling to a public-recorder append after ACK."""
    def __init__(self, session):
        self.session = session

    @property
    def current_public_observation(self):
        return self.session.current_public_observation

    def update(self, frame, *, phase):
        # Session.__call__ has already stored this exact acknowledged frame.
        self.session.phases[-1] = str(phase)
        self.session.action_status[-1]['phase'] = str(phase)


def _observation_loop(mode, initial, rpc, root, horizon, actors, record):
    controller = importlib.import_module('tasks.' + mode + '.controller')
    if mode == 'insert_key':
        os.environ['IK_RESULT'] = str(root / 'policy')
        actor = controller.Model()
        actor.reset()
    else:
        from harness.runtime.external_rule_assets import portable_coin_assets, resource
        task = ROOT / 'tasks' / mode
        assets = portable_coin_assets(task, root / 'policy-assets')
        from tasks.deposit_coin.skills.perception.calibration import visual_context
        defaults = visual_context()
        actor = controller.RuleSkill(assets, resource(task, 'x5-collision.yml'), defaults,
                                    stop_after=None, transfer='handoff')
    actors.append(actor)
    frame, count = initial, 0
    while count < horizon:
        if mode == 'insert_key':
            actor.update_obs_batch([dict(color_frame(frame), env_idx=0)])
            raw = actor.get_action_batch([0])[0][0]
            actions = [normalized_action({side + suffix: raw[side + original]
                for side in ('left', 'right')
                for suffix, original in (('_ee_pose', '_ee_pose'), ('_ee_joint_state', '_gripper'))})]
        else:
            protocol = importlib.import_module('tasks.deposit_coin.skills.observation.implementation')
            public = color_frame(frame)
            observation = protocol.Observation.create(count, count, round(count/(25*.004)),
                {name: public['vision'][name]['color'] for name in protocol.CAMERAS}, public['state'])
            chunk = actor.decide(observation)
            if chunk.observation_id != count or chunk.start_control_step != count or len(chunk.actions) != 16:
                raise RuntimeError('Original coin chunk identity or length changed')
            actions = chunk.actions
        for action in actions:
            if count >= horizon:
                break
            frame = rpc(normalized_action(action))
            count += 1
            record['acknowledged_actions'] = count


def _stack(mode, source, resources, initial, rpc, root, horizon, actors, args):
    robot = source.robot
    from harness.runtime.native_rule_evaluation.binding import robot_configurations, fk, action_submission
    from skills.shared.planning.declared_x5_numeric_tools.implementation import create
    numeric = create(resources, root / 'numeric-tools')
    configurations = robot_configurations(resources, robot)
    evidence = robot.PublicRecorder(root)
    session = PublicSession(initial, rpc, root, horizon, evidence, robots=configurations, task_name=mode)
    env = session.environment
    env.robot_manager = numeric
    diagnostics = source.planner.configure_planner_table(env)
    planner_class = getattr(robot, 'PlannerTools', getattr(robot, 'PublicPlanner', None))
    as_numpy = importlib.import_module('tasks.' + mode + '.skills.motion_math.implementation').as_numpy
    planners = {name: planner_class(planner.plan_path, planner.plan_joint,
        partial(fk, planner, as_numpy=as_numpy)) for name, planner in numeric.planner.items()}
    manager = robot.PublicRobotManager(configurations if mode == 'stack_bowls_random' else configurations.values(), planners, evidence)
    environment = robot.PublicEnvironment(manager, .004, horizon)
    clock = PublicCommandClock(env, _ClockHistory(session), lambda action: session(action)[0], time_scale=1.8)
    # Every stack class keeps its existing submission type; bowls retains its
    # phase-specific loaded-motion time_scale setters on this same clock.
    submission = action_submission(clock, robot) if mode == 'stack_bowls_random' else robot.ActionSubmission(clock.execute)
    actor = source.context(environment, evidence, args, submission, diagnostics)
    actors.append(actor)
    return actor, session


def run(mode, declaration, initial, exchange, output_root, *, prepare_runtime=prepare, public_parameters=None):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    family = declaration['adapter_family']
    if family not in FAMILIES:
        raise ValueError('Unsupported public task adapter family')
    horizon = int(declaration['official_action_horizon'])
    actors, source, session = [], None, None
    record = {'policy_error': None, 'traceback': None, 'adapter_family': family,
              'official_success_claimed': False, 'official_score_assigned': False,
              'action_count_is_acknowledged_only': True}
    try:
        if family != 'public_observation_policy':
            source = importlib.import_module('tasks.' + mode + '.bindings.official_policy')
        prepare_runtime(mode, source)
        if family == 'public_observation_policy':
            return _observation_loop(mode, initial, exchange, root, horizon, actors, record)
        args = SimpleNamespace(motion_speed=getattr(source, 'MOTION_SPEED', 1.8), motion_speedup=1.,
            planner_pose_frame='world', gripper_forward_offset=.145, feedback_servo=False,
            debug_fk=False, official_eval=True, perception_only=False, no_video=False,
            images_in_hdf5=False)
        if public_parameters:
            if mode != 'sort_nesting_dolls_by_size' or public_parameters != {'public_shape_profile': 'random'}:
                raise ValueError('Unsupported public geometry profile')
            args.public_shape_profile = 'random'
        task = ROOT / 'tasks' / mode
        if family == 'public_stack_workflow':
            resources = source.binding.verify_resources(task)
            actor, session = _stack(mode, source, resources, initial, exchange, root, horizon, actors, args)
            return actor.run()
        if family == 'public_toast_workflow':
            from harness.runtime.public_policy_recording import PublicEvidence, verify_resources
            resources = verify_resources(task)
            evidence = PublicEvidence(root)
        else:
            resources = source.verify_resources(task)
            evidence = source.PublicEvidence(root)
        session = PublicSession(initial, exchange, root, horizon, evidence, task_name=mode)
        env = session.environment
        if family == 'public_toast_workflow' or mode.startswith('arrange_largest_number') or mode == 'sort_nesting_dolls_by_size':
            from skills.shared.planning.declared_x5_numeric_tools.implementation import create
            env.robot_manager = create(resources, root / 'numeric-tools',
                arm_joints_only=mode.startswith('arrange_largest_number'))
        elif mode == 'stack_blocks_by_language':
            # This factory already constructs its own public planners and
            # installs their fixed table. Its old compatibility call also
            # visits the otherwise-unused native planner collection. There
            # is no such collection in a websocket policy process.
            env.robot_manager = SimpleNamespace(planner={})
        source.run(env, session, resources, session, args, actors)
    except PublicActionBudgetReached:
        record['own_command_budget_reached'] = True
    except Exception as error:
        record.update(policy_error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        print(json.dumps({'event': 'policy_bug_enter_finite_hold', 'error': record['policy_error']}), flush=True)
        traceback.print_exc()
    finally:
        if session is not None:
            record['acknowledged_actions'] = int(session.environment.take_action_cnt[0])
            record['phases'] = session.action_status
        diagnostics = []
        for actor in actors:
            try:
                diagnostics.append(source.diagnostics(actor) if source is not None and hasattr(source, 'diagnostics')
                    else actor.result() if hasattr(actor, 'result') else {'public_class': type(actor).__name__})
            except Exception as error:
                diagnostics.append({'diagnostics_error': repr(error)})
        record['actors'] = diagnostics
        save(root / 'policy-diagnostics.json', record)
        if mode == 'sort_nesting_dolls_by_size':
            # Its original bound collision LRU retains whole contexts (and
            # their owned CUDA tools). No later episode may keep that owner.
            # Within-episode caching and every numerical result are unchanged.
            for actor in actors:
                method = getattr(actor, 'public_collision_sections', None)
                if method is not None:
                    method.cache_clear()
