"""Acquire calibrated wrist silhouettes before estimating bowl dimensions."""
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.support_observation import observe_support_geometry
from physicalrsi_observed_geometry.support_projection import support_point
from physicalrsi_observed_geometry.supported_frustum import fit_supported_frustum
from physicalrsi_observed_geometry.view_acquisition import wrist_look_at
from physicalrsi_skill_library.geometry import matrix, forward_kinematics
from perception import segment_many_http, detect_bowls
from geometry import finger_clearance

CONTROLS = ['left_arm_joint_state', 'right_arm_joint_state', 'left_ee_joint_state', 'right_ee_joint_state']


def plan_view(owner, arm, proposal):
    from model import get_ik
    from rdj_insert_policy.ik import observed_base
    ik = get_ik(owner).solver
    observation = owner.obs
    view = wrist_look_at(observation, arm, proposal)
    initial = np.asarray(observation['state'][arm+'_ee_pose'])
    target = np.asarray(view['pose'])
    seed = np.asarray(observation['state'][arm+'_arm_joint_state'])
    original = seed.copy()
    base = observed_base(ik.chain, arm, observation['state'])
    interpolation = Slerp([0, 1], Rotation.from_quat(np.array([initial[[4,5,6,3]], target[[4,5,6,3]]])))
    points = []
    minimum = float('inf')
    for fraction in np.linspace(0, 1, 17)[1:]:
        pose = np.r_[initial[:3], interpolation(fraction).as_quat()[[3,0,1,2]]]
        gap = finger_clearance(matrix(pose), support_plane=owner.observed_support_plane)
        if gap < .005:
            raise GeometryUnavailable('Inspection finger envelope approaches observed support')
        q = ik.solve(arm, seed, pose, observed_state=observation['state'],
                     support=owner.observed_support, env_idx=owner.env_idx)
        if q is None:
            raise GeometryUnavailable('Inspection IK or public arm mesh check rejected')
        q = np.asarray(q)
        if np.max(np.abs(q-seed)) > .3:
            raise GeometryUnavailable('Inspection would switch joint branch')
        for sample in np.linspace(seed, q, 5):
            actual = forward_kinematics(ik.chain, base, sample)
            gap = finger_clearance(matrix(actual), support_plane=owner.observed_support_plane)
            minimum = min(minimum, gap)
            if gap < .005 or not ik.mesh_collision.is_collision_free(sample):
                raise GeometryUnavailable('Sampled inspection path collides with arm or support')
        points.append(q.tolist()); seed = q
    return {'arm': arm, 'points': points, 'original': original.tolist(),
            'pose': view['pose'], 'rotation_radians': view['rotation_radians'],
            'minimum_finger_support_gap_m': minimum,
            'scope': 'Sampled self-mesh/support checks; not an inter-arm or task-object collision certificate'}


def match_bowls(head, rows, head_masks, wrist, wrist_masks, plane):
    resolved = {}
    for row in rows:
        fits = []
        for mask in wrist_masks:
            try:
                fit = fit_supported_frustum([(head, head_masks[row['index']]), (wrist, mask)], plane)
            except GeometryUnavailable:
                continue
            fits.append(fit)
        if not fits:
            continue
        fits.sort(key=lambda fit: fit['rms_px'])
        first = fits[0]
        # A competing object association must not silently choose a different bowl.
        if any(np.linalg.norm(fit['base']-first['base']) > .01 or
               abs(fit['top_radius']-first['top_radius']) > .1*first['top_radius'] for fit in fits[1:]):
            continue
        rim = first['base'] + first['height']*first['rotation'][:, 2]
        resolved[row['color']] = {'base': first['base'].tolist(), 'height': float(first['height']),
            'radius': float(first['top_radius']), 'rim_center': rim.tolist(),
            'rotation': first['rotation'].tolist(), 'rms_px': float(first['rms_px']),
            'relative_spread': first['relative_spread'].tolist(),
            'source': 'current_head_and_acquired_wrist_RGB_with_observed_support',
            'shape_hypothesis': 'upright_circular_frustum', 'hidden_geometry_verified': False}
    return resolved


class BowlAcquisition:
    def __init__(self):
        self.phase = 'initialize'
        self.resolved = {}
        self.attempted = set()
        self.steps = 0
        self.station_steps = 0
        self.out = Path(os.environ['PBL_DIAGNOSTICS']) / ('inspection-'+str(time.time_ns()))
        self.out.mkdir(parents=True)

    def record(self, **row):
        with (self.out/'trace.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': self.steps, 'phase': self.phase, **row})+'\n')

    def hold(self, owner):
        return [{key: np.asarray(owner.obs['state'][key]).reshape(-1).tolist() for key in CONTROLS}]

    def initialize(self, owner):
        self.head = Camera.from_observation(owner.obs)
        group = segment_many_http(self.head.image, [('bowl', .5)])[0]
        self.rows = detect_bowls(self.head.image, self.out/'head', group=group)
        self.head_masks = [np.asarray(mask, bool).copy() for mask in group.masks]
        support = observe_support_geometry(owner.obs, segment_many_http, occluders=('robot arm', 'bottle', 'bowl'))
        owner.observed_support = support['patches']
        owner.observed_support_plane = support['plane']
        self.record(event='observed_support', plane=np.asarray(support['plane']).tolist())
        self.next_view(owner)

    def next_view(self, owner):
        pending = [row for row in self.rows if row['color'] not in self.resolved]
        if not pending:
            owner.bowl_observations = self.resolved
            self.phase = 'complete'
            self.record(event='complete', bowls=self.resolved)
            return
        row = pending[0]
        proposal = support_point(row['centroid'], 0, camera=self.head, support_plane=owner.observed_support_plane)
        choices = []
        for arm in ('left', 'right'):
            if (row['color'], arm) in self.attempted:
                continue
            try:
                view = wrist_look_at(owner.obs, arm, proposal)
                choices.append((view['rotation_radians'], arm))
            except GeometryUnavailable as error:
                self.attempted.add((row['color'], arm))
                self.record(event='view_rejected', bowl=row['color'], arm=arm, reason=str(error))
        for _, arm in sorted(choices):
            self.attempted.add((row['color'], arm))
            try:
                plan = plan_view(owner, arm, proposal)
            except GeometryUnavailable as error:
                self.record(event='path_rejected', bowl=row['color'], arm=arm, reason=str(error))
                continue
            self.plan = plan
            self.path = plan['points']
            self.index = 0
            self.station_steps = 0
            self.settle = 0
            self.phase = 'view'
            self.record(event='planned', bowl=row['color'], **plan)
            return
        raise GeometryUnavailable('No qualified remaining wrist view for '+row['color'])

    def measure(self, owner):
        wrist = Camera.from_observation(owner.obs, 'cam_'+self.plan['arm']+'_wrist')
        group = segment_many_http(wrist.image, [('bowl', .5)])[0]
        rows = [row for row in self.rows if row['color'] not in self.resolved]
        found = match_bowls(self.head, rows, self.head_masks, wrist, group.masks, owner.observed_support_plane)
        self.resolved.update(found)
        self.record(event='measured', found=found)
        self.path = list(reversed(self.plan['points'][:-1])) + [self.plan['original']]
        self.index = 0
        self.station_steps = 0
        self.phase = 'return'

    def step(self, owner):
        if self.phase == 'complete':
            return None
        if self.phase == 'failed':
            return self.hold(owner)
        self.steps += 1
        try:
            if self.steps > 400:
                raise GeometryUnavailable('Bowl inspection exceeded its bounded motion budget')
            if self.phase == 'initialize':
                self.initialize(owner)
            if self.phase == 'complete':
                return None
            if self.phase == 'settle':
                self.settle += 1
                if self.settle >= 3:
                    actual = np.asarray(owner.obs['state'][self.plan['arm']+'_ee_pose'])
                    expected = np.asarray(self.plan['pose'])
                    angle = (Rotation.from_quat(actual[[4,5,6,3]]).inv()*Rotation.from_quat(expected[[4,5,6,3]])).magnitude()
                    if np.linalg.norm(actual[:3]-expected[:3]) > .005 or angle > .04:
                        raise GeometryUnavailable('Inspection pose did not converge in observed proprioception')
                    self.measure(owner)
                return self.hold(owner)
            arm = self.plan['arm']
            current = np.asarray(owner.obs['state'][arm+'_arm_joint_state'])
            target = np.asarray(self.path[self.index])
            delta = target-current
            if np.max(np.abs(delta)) < .008:
                self.index += 1
                self.station_steps = 0
                if self.index == len(self.path):
                    if self.phase == 'view':
                        self.phase = 'settle'; self.settle = 0
                    else:
                        self.next_view(owner)
                    return self.hold(owner) if self.phase != 'complete' else None
                target = np.asarray(self.path[self.index]); delta = target-current
            self.station_steps += 1
            if self.station_steps > 20:
                raise GeometryUnavailable('Inspection joint waypoint did not converge')
            action = self.hold(owner)[0]
            action[arm+'_arm_joint_state'] = (current+delta*min(1., .06/max(float(np.max(np.abs(delta))), 1e-9))).tolist()
            return [action]
        except (GeometryUnavailable, ValueError) as error:
            self.phase = 'failed'
            self.record(event='rejected', reason=str(error))
            return self.hold(owner)
