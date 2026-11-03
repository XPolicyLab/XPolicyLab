"""Nominal placed-tube obstacles derived from current RGB fits and planned poses.

Checks sampled arm paths with public finger opening envelopes, not carried tubes,
inter-arm collisions, or uncertainty in the achieved placement. No fixture geometry.
"""
from itertools import permutations
import numpy as np
from scipy.spatial.transform import Rotation


def transform(pose):
    p = np.asarray(pose, float)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError('Finite observed pose required')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = p[:3]
    return result


class PlacedTubeScene:
    def __init__(self, guard, bases, cylinders):
        import hppfcl
        import trimesh
        self.fcl = hppfcl
        self.guard = guard
        self.bases = {arm: transform(pose) for arm, pose in bases.items()}
        self.obstacles = {}
        for key, (radius, length, pose) in cylinders.items():
            if not np.isfinite([radius, length]).all() or min(radius, length) <= 0:
                raise ValueError('Positive RGB-fitted cylinder dimensions required')
            world = transform(pose)
            self.obstacles[key] = (hppfcl.Cylinder(radius, length),
                hppfcl.Transform3f(world[:3, :3], world[:3, 3]),
                world[:3, 3], float(np.hypot(radius, length / 2)))
        self.links = []
        for i, obj in enumerate(guard.geometry.geometryObjects):
            if obj.parentJoint not in {0, *guard.joint_ids}:
                continue
            vertices = np.asarray(trimesh.load(obj.meshPath, force='mesh').vertices) * obj.meshScale
            self.links.append((i, float(np.linalg.norm(vertices, axis=1).max())))
        self.active = ()

    def path(self, arm, start, targets, resolution=.05):
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('Positive joint sampling resolution required')
        guard = self.guard
        previous = np.asarray(start, float)
        result = []
        for target in targets:
            target = np.asarray(target, float)
            if (target.shape != (len(guard.joint_ids),) or previous.shape != target.shape
                    or not np.isfinite(np.r_[target, previous]).all()):
                raise ValueError('Finite arm joint path required')
            count = max(1, int(np.ceil(np.max(abs(target - previous)) / resolution)))
            for fraction in np.linspace(0., 1., count + 1):
                q = guard.pin.neutral(guard.model)
                for joint, value in zip(guard.joint_ids, previous + fraction * (target - previous)):
                    q[guard.model.joints[joint].idx_q] = value
                guard.pin.updateGeometryPlacements(guard.model, guard.data,
                    guard.geometry, guard.geometry_data, q)
                result.append([(i, radius, self.bases[arm] @ guard.geometry_data.oMg[i].homogeneous)
                               for i, radius in self.links])
            previous = target
        return result

    def clear(self, path, keys):
        for frame in path:
            for i, radius, world in frame:
                for key in keys:
                    shape, obstacle, center, bound = self.obstacles[key]
                    if np.linalg.norm(world[:3, 3] - center) > radius + bound:
                        continue
                    result = self.fcl.CollisionResult()
                    self.fcl.collide(self.guard.geometry.geometryObjects[i].geometry,
                        self.fcl.Transform3f(world[:3, :3], world[:3, 3]), shape, obstacle,
                        self.fcl.CollisionRequest(), result)
                    if result.isCollision():
                        return False
        return True

    def segment_clear(self, arm, start, target):
        return not self.active or self.clear(self.path(arm, start, [target]), self.active)


def choose_order(groups, path_for, clear_pair, minimum_spacing):
    """Maximize placements then minimize cost, retaining distinct pose alternatives."""
    best = None
    paths, compatible = {}, {}
    groups = [tuple(sorted(group, key=lambda c: c.cost)) for group in groups]
    for size in range(len(groups), 0, -1):
        for order in permutations(range(len(groups)), size):
            def visit(chosen, cost):
                nonlocal best
                if len(chosen) == size:
                    if best is None or cost < best[0]:
                        best = (cost, tuple(chosen))
                    return
                for candidate in groups[order[len(chosen)]]:
                    if best is not None and cost + candidate.cost >= best[0]:
                        break
                    if any(candidate.slot_index == old.slot_index or
                           np.linalg.norm(np.asarray(candidate.slot_position) - old.slot_position) < minimum_spacing
                           for old in chosen):
                        continue
                    valid = True
                    for old in chosen:
                        key = (id(candidate), old.label, old.slot_index)
                        if key not in compatible:
                            if id(candidate) not in paths:
                                paths[id(candidate)] = path_for(candidate)
                            compatible[key] = clear_pair(paths[id(candidate)], old)
                        if not compatible[key]:
                            valid = False
                            break
                    if valid:
                        visit(chosen + [candidate], cost + candidate.cost)
            visit([], 0.)
        if best is not None:
            return best[1]
    return ()


def observed_assignment(runtime, env_idx, groups, value, robot, stages, pickup_contacts,
                        delivery_poses, minimum_spacing, observed_base, to_ee):
    tubes, lengths = value['tube_poses'], value['tube_lengths']
    from .delivery_observability import observable_candidates
    before = [len(group) for group in groups]
    groups = observable_candidates(groups, runtime.observations[env_idx], tubes, lengths,
                                   pickup_contacts, delivery_poses, to_ee)
    print('[insert_observable_candidates]', env_idx, before, '->', [len(group) for group in groups], flush=True)
    support = runtime.perception.support_geometry(env_idx)
    if not support:
        raise ValueError('Observed support required for resting tube radius')
    plane = transform(support[0]['pose'])
    normal = plane[:3, 2]
    level = normal @ plane[:3, 3] + support[0]['dims'][2] / 2
    cylinders = {}
    for group in groups:
        for candidate in group:
            i = int(candidate.label.removeprefix('tube'))
            radius = float(normal @ np.asarray(tubes[i][:3]) - level)
            length = float(lengths[i])
            pickup = transform(pickup_contacts(tubes[i], length)[candidate.pickup_index])
            guide = transform(delivery_poses(tubes[i], pickup_contacts(tubes[i], length)[candidate.pickup_index], candidate.slot_pose)['guide'])
            attachment = pickup[:3, :3].T @ (np.asarray(tubes[i][:3]) - pickup[:3, 3])
            slot = transform(candidate.slot_pose)
            root = guide[:3, 3] + guide[:3, :3] @ attachment
            center = root + slot[:3, 2] * length / 2
            cylinders[(candidate.label, candidate.slot_index)] = (radius, length,
                center.tolist() + list(candidate.slot_pose[3:]))
    state = runtime.observations[env_idx]['state']
    bases = {arm: observed_base(runtime.ik.chain, arm, state) for arm in ('left', 'right')}
    from .finger_envelope import FingerEnvelopeScene
    scene = FingerEnvelopeScene(runtime.ik.mesh_collision, bases, cylinders)
    def path_for(candidate):
        targets = [stage['target'][candidate.arm + '_arm_joint_state']
                   for stage in stages(robot, [candidate])]
        return scene.path(candidate.arm, state[candidate.arm + '_arm_joint_state'], targets)
    selected = choose_order(groups, path_for,
        lambda path, old: scene.clear(path, [(old.label, old.slot_index)]), minimum_spacing)
    runtime.ik.placed_tube_scenes[env_idx] = scene
    return selected


def activate_completed_placements(runtime, env_idx, plan):
    scene = runtime.ik.placed_tube_scenes.get(env_idx)
    if scene is None:
        return
    elapsed, placed = 0, []
    for stage in plan['stages']:
        elapsed += int(stage['steps'])
        if elapsed > runtime.action_counts[env_idx]:
            break
        if stage['name'].endswith(':release'):
            label = stage['name'].split(':', 1)[0]
            placed.append((label, runtime.contexts[env_idx][label]['slot_index']))
    scene.active = tuple(placed)
