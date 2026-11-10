"""Control play_Xylophone with episode-specific RGB geometry."""

from collections import deque

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import matrix_quat, pose_matrix
from .observed_xylophone_vision import (
    associated_head, board_depth, held_head, keyboard, mallet, mallet_depth,
    on_plane, rays, views_for_observation,
)
from .policy import Policy, Waypoint
from .xylophone_connected import ConnectedArm
from .xylophone_feedback import ExecutionWindow
from .xylophone_routes import JOINT_MARGIN, PROBE_PATTERNS, MotionProgress
from .xylophone_x37 import XylophonePolicy as ContinuousReplanning
from .xylophone_x41 import XylophonePolicy as ContinuousMotion


def transform_point(frame, point):
    return frame[:3, 3]+frame[:3, :3]@point


class StrokeProgress:
    """Track whether a commanded descent stopped near the measured key."""

    def __init__(self, radius):
        self.radius = radius
        self.clear_window()

    def clear_window(self):
        self.name = None
        self.points = deque(maxlen=6)
        self.top = None

    def observe(self, name, current, goal, anchor):
        if not name.endswith('_hit'):
            self.clear_window()
            return {'qualified': False}
        point = transform_point(pose_matrix(current), anchor)
        target = transform_point(pose_matrix(goal), anchor)
        if self.name != name:
            self.clear_window()
            self.name = name
            self.top = point[2]
        self.points.append(point)
        displacement = point-target
        descent = self.top-point[2]
        spread = np.linalg.norm(np.ptp(self.points, axis=0))
        close = np.linalg.norm(displacement[:2]) < .55*self.radius and 0 < displacement[2] < 1.5*self.radius
        return dict(qualified=bool(len(self.points) == self.points.maxlen and spread < .0015
                                   and descent > 1.0*self.radius and close),
                    descent_m=float(descent), head_error_m=displacement.tolist())


class ObservedXylophone(ContinuousMotion):
    """Use measured targets while retaining the original motion timing."""

    def __init__(self, calibration, use_servo=False):
        self.use_servo = use_servo
        super().__init__('play_Xylophone', calibration)

    def reset(self):
        Policy.reset(self)
        self.stage = 'observe'
        self.state_now = {}
        self.arms = {side: ConnectedArm.public_x5(side) for side in ('left', 'right')}
        self.side = self.anchor = self.nominal_anchor = self.grasp_pose = self.keys = None
        self.binding = {}
        self.gripper_targets = {}
        self.motion_progress = MotionProgress()
        self.motion_recoveries = []
        self.motion_retry_count = self.key_replans = self.unreachable_advances = 0
        self.lowered_hovers = set()
        self.feedback_events = []
        self.execution = ExecutionWindow()
        self.last_reached_pose = None
        self.keyboard_refreshed = False
        self.forced_side = None
        self.grasp_sides_tried = []
        self.board = self.key_surface = self.radius = self.strike_z = self.hover_z = None
        self.rgb_samples = []
        self.program_notes = {'candidate': 'play_Xylophone'}

    def camera_views(self):
        return views_for_observation(self.perception.camera, dict(state=self.state_now, vision=self.current_vision))

    def _observe(self, rgb):
        self.attempts += 1
        camera = self.perception.camera
        table = self.calibration['table_z']
        board = keyboard(camera, rgb, table)
        tool = mallet(rgb)
        if board is None or tool is None:
            self.reason = 'xy_observe_color_edges_and_shaft'
            return
        wrist = [entry for name, entry in self.camera_views().items() if name != 'cam_head']
        depth = board_depth(camera, board, wrist, table)
        if depth is None:
            self.reason = 'xy_observe_board_parallax'
            return
        surface, quality = depth
        self.board = board
        self.key_surface = surface
        self.keys = on_plane(camera, board['pixels'], surface)[:, :2]
        center_on_board = on_plane(camera, tool['center'], surface)
        apparent = tool['radius_px']/camera.k[0, 0]
        estimated_radius = float(apparent*np.linalg.norm(center_on_board-camera.c2w[:3, 3]))
        stereo = mallet_depth(camera, tool, wrist, table, board['pitch'])
        if stereo is not None and stereo[0][2] >= surface:
            head, self.radius = stereo
            origin = 'resting_multiview_rays'
        else:
            self.radius = estimated_radius
            # The board paint plane gives a scene scale for the silhouette
            # when the resting head is outside the two wrist fields of view.
            head = on_plane(camera, tool['center'], surface + 1.15*self.radius)
            origin = 'resting_silhouette_and_neighbor_plane'
        pinch_pixel = tool['center'] + .26*tool['extent']*tool['axis']
        grasp = on_plane(camera, pinch_pixel, head[2]-.28*self.radius)
        axis = grasp[:2]-head[:2]
        axis /= np.linalg.norm(axis)
        tip_pixel = tool['center'] + tool['extent']*tool['axis']
        tip = on_plane(camera, tip_pixel, grasp[2])
        normal = np.array([-tool['axis'][1], tool['axis'][0]])
        rails = on_plane(camera, pinch_pixel + np.array([-.5, .5])[:, None]*2*tool['half_width_px']*normal, grasp[2])
        self.shaft_span = float(np.linalg.norm(tip-grasp) + np.linalg.norm(grasp-head))
        self.shaft_half_width = float(np.linalg.norm(rails[1]-rails[0])*.5)
        self.program_notes['rgb_scene'] = dict(visible_keys=int(board['observed_bars']),
            paint_plane=float(surface), plane_match=quality, radius_m=float(self.radius),
            resting_depth_source=origin, grasp_height_m=float(grasp[2]),
            visible_shaft_length_m=self.shaft_span, visible_shaft_radius_m=self.shaft_half_width)
        self.last_detections = dict(head_xy=head[:2].tolist(), shaft_axis=axis.tolist(),
                                    keys_xy=self.keys.tolist())
        self.hit_stall = StrokeProgress(self.radius)
        # Color alignment is influenced by the painted sidewalls. Account for
        # the observed silhouette scale in the target before the rebound.
        paint_depth_uncertainty = .22*board['pitch']
        self.strike_z = surface + paint_depth_uncertainty + 1.35*self.radius
        self.hover_z = self.strike_z + max(4.1*self.radius, 1.18*board['pitch'])
        self.plan_pickup(grasp, axis, head)

    def plan_pickup(self, point, axis, head):
        preferred = self._choose_side(point[:2], self.keys)
        yaws = [np.arctan2(axis[1], axis[0]), np.arctan2(axis[1], axis[0])+np.pi]
        for side in (preferred, 'right' if preferred == 'left' else 'left'):
            self.side = side
            candidates = []
            for yaw_index, yaw in enumerate(yaws):
                grab = self.pose(side, point[:2], point[2]-self.calibration['table_z'], yaw)
                frame = pose_matrix(grab)
                anchor = frame[:3, :3].T@(head-frame[:3, 3])
                approach = grab.copy()
                approach[2] += .09
                lift = grab.copy()
                lift[2] += .065
                prefix = [Waypoint(side, approach, 1., 'xy_approach', 2),
                          Waypoint(side, grab, 1., 'xy_descend', 3),
                          Waypoint(side, grab, 0., 'xy_close', 10),
                          Waypoint(side, lift, 0., 'xy_probe_0', 3)]
                return_path = [Waypoint(side, lift, 0., 'xy_return_above', 2),
                               Waypoint(side, grab, 0., 'xy_return_lower', 3),
                               Waypoint(side, grab, 1., 'xy_release', 8),
                               Waypoint(side, approach, 1., 'xy_retreat', 2)]
                for probe_index, offsets in enumerate(PROBE_PATTERNS):
                    probes = []
                    for index, offset in enumerate(offsets):
                        probe = lift.copy()
                        probe[:3] += offset
                        probes.append(Waypoint(side, probe, 0., f'xy_probe_{index+1}', 3))
                    plan = prefix+probes
                    if self._audit(side, plan+return_path, anchor, reserve=70, local_only=True):
                        audit = self.program_notes['path_audit']
                        rank = self._rank_grasp(audit, len(PROBE_PATTERNS)*yaw_index+probe_index)
                        candidates.append((rank, plan, grab, anchor))
                if not candidates and self._audit(side, prefix+return_path, anchor, reserve=70, local_only=True):
                    compact_probe = prefix[-1].pose.copy()
                    compact_probe[:2] += .65*self.board['pitch']*axis
                    candidates.append(((2, 0, yaw_index), prefix+[Waypoint(side, compact_probe, 0., 'xy_probe_1', 3)],
                                       grab, anchor))
            if candidates:
                _, plan, self.grasp_pose, self.nominal_anchor = min(candidates, key=lambda item: item[0])
                self.rgb_samples = []
                self._set_plan(plan, 'bind')
                self.reason = 'xy_rgb_grasp_and_probe'
                return
        self.reason = 'xy_no_continuous_pickup_route'

    def optical_samples(self, current, prior):
        ee = pose_matrix(current)
        predicted = transform_point(ee, prior)
        result = []
        for camera, image in self.camera_views().values():
            depth = (predicted-camera.c2w[:3, 3])@(-camera.c2w[:3, 2])
            if depth <= 0:
                continue
            match = associated_head(camera, image, predicted, search_fraction=.13)
            if match is None:
                continue
            pixel, radius = match
            local_camera = type(camera)(dict(K=camera.k,
                camera_to_world_opengl=np.linalg.inv(ee)@camera.c2w,
                resolution=camera.resolution))
            result.append((local_camera, pixel, radius))
        return result

    def _record_probe(self, rgb, current, force=False):
        if self.nominal_anchor is None or (not force and self.step % 4):
            return
        self.rgb_samples.extend(self.optical_samples(current, self.nominal_anchor))

    def _bind(self):
        measured = held_head(self.rgb_samples)
        if measured is not None:
            anchor, radius, error = measured
            if (error <= 2. and .6*self.radius < radius < 1.65*self.radius
                    and np.linalg.norm(anchor-self.nominal_anchor) < 3.*self.radius):
                self.anchor = anchor
                self.radius = .5*(self.radius+radius)
                self.binding = dict(state='consistent', source='multiview_observed_held_head',
                                    residual_px=error, samples=len(self.rgb_samples))
        if self.anchor is None:
            frame = pose_matrix(self.grasp_pose)
            direction_of_sag = frame[:3, :3].T@np.array([0., 0., -1.])
            prior = self.nominal_anchor + .85*self.radius*direction_of_sag
            estimates = []
            for camera, pixel, size in self.rgb_samples:
                ray = rays(camera, pixel)
                ray /= np.linalg.norm(ray)
                distance = (prior-camera.c2w[:3, 3])@ray
                estimate = camera.c2w[:3, 3] + distance*ray
                if np.linalg.norm(estimate-prior) < 1.4*self.radius:
                    estimates.append(estimate)
            self.anchor = (.88*prior + .12*np.median(estimates, axis=0)) if estimates else prior
            self.binding = dict(state='propagated', source='measured_resting_geometry_with_sag_model',
                                lateral_views=len(estimates), samples=len(self.rgb_samples))
        self.program_notes['observed_anchor'] = self.anchor.tolist()
        self.hit_stall = StrokeProgress(self.radius)
        self.stage = 'keyboard'

    def _refresh_keyboard(self, rgb):
        self.keyboard_refreshed = True
        return True

    def _endpoint_checked_strokes(self, side, anchor, prefix, key_indices=None):
        arm = self.arms[side]
        seed = self.state_now[side+'_arm_joint_state'].copy()
        current = pose_matrix(self.state_now[side+'_ee_pose'])
        center = transform_point(current, anchor)
        targets = [('xy_transfer_functional_raise', np.r_[center[:2], max(self.hover_z, center[2])])]
        indices = range(len(self.keys)) if key_indices is None else key_indices
        for key in indices:
            for phase, height in (('hover', self.hover_z), ('hit', self.strike_z), ('lift', self.hover_z)):
                targets.append((f'xy_key_{key}_{phase}', np.r_[self.keys[key], height]))
        route = []
        tcp = self.calibration['tool_offset_in_ee']
        tip = self.calibration['contact_tip_offset_in_ee']
        table = self.calibration['table_z']
        shaft = np.asarray(tcp)-anchor
        shaft /= np.linalg.norm(shaft)
        for name, point in targets:
            previous_rotation = arm.forward(seed)[:3, :3]
            lower = np.maximum(arm.lower+JOINT_MARGIN, seed-.9)
            upper = np.minimum(arm.upper-JOINT_MARGIN, seed+.9)

            def residual(q):
                frame = arm.forward(q)
                position = transform_point(frame, anchor)
                rotation = Rotation.from_matrix(previous_rotation.T@frame[:3, :3]).as_rotvec()
                shaft_z = (frame[:3, :3]@shaft)[2]
                shaft_low = position[2]+min(0., self.shaft_span*shaft_z)-self.shaft_half_width
                finger_low = transform_point(frame, tip)[2]
                clearance = min(shaft_low, finger_low)-table
                return np.r_[300.*(position-point), .002*rotation, .0005*(q-seed),
                             300.*max(0., .008-clearance)]

            fit = least_squares(residual, np.clip(seed, lower+1.e-8, upper-1.e-8),
                                bounds=(lower, upper), max_nfev=120,
                                ftol=1.e-10, xtol=1.e-10, gtol=1.e-10)
            seed = fit.x
            frame = arm.forward(seed)
            route.append(Waypoint(side, np.r_[frame[:3, 3], matrix_quat(frame[:3, :3])],
                                  0., name, 2, anchor))
        self.program_notes['stroke_geometry'] = dict(strike_height_m=float(self.strike_z),
                                                     lift_height_m=float(self.hover_z))
        return route

    def _choose_strokes(self, side, rotation, anchor, prefix=(), reserve=60):
        return self._endpoint_checked_strokes(side, anchor, prefix), dict(degrees=0., orientation='observed_horizontal')

    def _observe_stroke_head(self, waypoint, current):
        if not self.use_servo:
            return
        observations = self.optical_samples(current, self.anchor)
        if not observations:
            return
        shifts = []
        for camera, pixel, _ in observations:
            ray = rays(camera, pixel)
            ray /= np.linalg.norm(ray)
            origin = camera.c2w[:3, 3]
            shift = origin + ((self.anchor-origin)@ray)*ray - self.anchor
            if np.linalg.norm(shift) < self.radius:
                shifts.append(shift)
        if not shifts:
            return
        new = self.anchor + .36*np.median(shifts, axis=0)
        for remaining in self.plan[self.index+1:]:
            frame = pose_matrix(remaining.pose)
            position = transform_point(frame, self.anchor)-frame[:3, :3]@new
            remaining.pose[:3] = position
            remaining.anchor_offset = new
        self.anchor = new
        self.program_notes['wrist_feedback_updates'] = 1+self.program_notes.get('wrist_feedback_updates', 0)

    def _recover_motion(self, waypoint, current, action, evidence):
        # Replan from the robot's measured state at the same rebound height.
        ContinuousReplanning._recover_motion(self, waypoint, current, action, evidence)

    def _recover_stalled_hit(self, waypoint, current, anchor, action, evidence):
        contact = evidence.get('contact_guard', {})
        if contact.get('qualified'):
            raise_by = float(np.clip(.70*contact['head_error_m'][2]+.12*self.radius, 0., .58*self.radius))
            self.strike_z += raise_by
            self.hover_z += raise_by
            for remaining in self.plan[self.index+1:]:
                if remaining.name.startswith('xy_key_'):
                    remaining.pose[2] += raise_by
        action[waypoint.side+'_ee_pose'] = current.copy()
        self._advance(waypoint, current, 'optical_hit_descent_complete')
        self.execution = ExecutionWindow()
