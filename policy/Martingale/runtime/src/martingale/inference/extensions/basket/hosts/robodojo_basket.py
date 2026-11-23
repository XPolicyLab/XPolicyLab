# Copyright © 2026 Martingale. All Rights Reserved.
import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation
from martingale.inference.extensions.basket.hosts import robodojo_basket_shape as shape

def hand_pose(state, arm):
    p = np.asarray(state[f'{arm}_ee_pose'], float)
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    t[:3, 3] = p[:3]
    return t

def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]

def targets_for(views, masks, robots):
    targets = []
    for cam, view in views.items():
        handle = masks[cam]['handle'] & shape.dilate(masks[cam]['whole'], 3)
        body = masks[cam]['whole'] & ~handle
        for kind, raw, other in [('body', body, handle), ('handle', handle, body)]:
            target = raw[::2, ::2].astype('uint8')
            if kind == 'body' and target.any():
                rc = np.argwhere(target)[:, ::-1].astype('int32')
                target[:] = 0
                cv2.fillConvexPoly(target, cv2.convexHull(rc), 1)
            target = target.astype(bool)
            count = target.sum()
            if count < 12:
                continue
            valid = ~(robots[cam] | shape.dilate(other))[::2, ::2] | target
            targets.append((cam, kind, view, target, valid, min(1.0, np.sqrt(count / 250))))
    return targets

class BasketTrack:

    def __init__(self, body, handle, step, state, fit_score):
        self.pose = np.eye(4)
        self.pose[:3, :3] = Rotation.from_euler('z', body[2]).as_matrix()
        self.pose[:3, 3] = [body[0], body[1], shape.kin.TABLE_Z]
        self.body = (shape.box_vertices(body) - self.pose[:3, 3]) @ self.pose[:3, :3]
        self.handle = (shape.arch_points(handle) - self.pose[:3, 3]) @ self.pose[:3, :3]
        crown = np.array([handle[0], handle[1], handle[4] + handle[5]])
        self.crown = (crown - self.pose[:3, 3]) @ self.pose[:3, :3]
        self.tangent = np.array([np.cos(handle[2]), np.sin(handle[2]), 0]) @ self.pose[:3, :3]
        self.radius = float(handle[-1])
        self.size = np.asarray(body[3:6])
        self.step = self.seen_step = step
        self.state = state
        self.velocity = np.zeros(3)
        self.velocity_known = False
        self.held_by = None
        self.attachment = None
        self.free_3d = False
        self.fit_score = float(fit_score)
        self.support = 2
        self.reason = 'initialized from multiple views'
        self.innovation = 0.0

    def predict(self, step, state):
        if self.held_by:
            return hand_pose(state, self.held_by) @ self.attachment
        result = self.pose.copy()
        result[:3, 3] += self.velocity * np.clip(step - self.step, 0, 20)
        return result

    def set_holder(self, arm):
        if arm == (self.held_by or ''):
            return
        if arm:
            self.attachment = np.linalg.inv(hand_pose(self.state, arm)) @ self.pose
            self.free_3d = True
        else:
            self.attachment = None
        self.held_by = arm or None
        self.velocity[:] = 0
        self.velocity_known = False

    def render(self, pose, view, kind):
        points = transform(self.body if kind == 'body' else self.handle, pose)
        rr, cc, depth = view.project(points)
        mask = np.zeros(shape.SHAPE, 'uint8')
        if np.any(depth <= 0.01):
            return mask.astype(bool)
        uv = np.rint(np.c_[cc, rr] / 2).astype('int32')
        if kind == 'body':
            cv2.fillConvexPoly(mask, cv2.convexHull(uv), 1)
        else:
            for i in range(len(uv) - 1):
                width = max(1, round(self.radius * view.fx / np.mean(depth[i:i + 2])))
                cv2.line(mask, tuple(uv[i]), tuple(uv[i + 1]), 1, width)
        return mask.astype(bool)

    def update(self, views, masks, robots, step, state):
        if step <= self.seen_step:
            return
        self.seen_step = step
        targets = targets_for(views, masks, robots)
        self.support = len({t[0] for t in targets})
        if self.support < 2:
            self.reason = 'occluded: prediction only; need two supporting cameras'
            return
        predicted = self.predict(step, state)
        dt = step - self.step
        reach = min(0.16, max(0.025, 0.006 * dt))
        bounds = [(-reach, reach), (-reach, reach)]
        if self.free_3d:
            bounds += [(-0.04, 0.04), (-0.22, 0.22), (-0.22, 0.22), (-0.22, 0.22)]
        else:
            bounds += [(-0.2, 0.2)]

        def pose_at(delta):
            pose = predicted.copy()
            if self.free_3d:
                pose[:3, 3] += delta[:3]
                pose[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ predicted[:3, :3]
            else:
                pose[:2, 3] += delta[:2]
                pose[:3, :3] = Rotation.from_euler('z', delta[2]).as_matrix() @ predicted[:3, :3]
            return pose

        def scores(pose):
            result = []
            for cam, kind, view, target, valid, weight in targets:
                mask = self.render(pose, view, kind) & valid
                result.append(2 * (mask & target).sum() / max(1, mask.sum() + target.sum()))
            return np.asarray(result)
        weights = [t[-1] for t in targets]

        def cost(delta):
            return 1 - np.average(scores(pose_at(delta)), weights=weights)
        zero = np.zeros(len(bounds))
        fit = differential_evolution(cost, bounds, x0=zero, seed=17, popsize=6, maxiter=22, tol=0.003, polish=False)
        refined = minimize(cost, fit.x, method='Powell', bounds=bounds, options={'maxiter': 12, 'xtol': 0.0003, 'ftol': 0.0001})
        delta = min([zero, fit.x, refined.x], key=cost)
        candidate = pose_at(delta)
        agreement = scores(candidate)
        score = float(np.average(agreement, weights=weights))
        cameras = {t[0] for t, s in zip(targets, agreement) if s >= 0.5}
        innovation = float(np.linalg.norm(candidate[:3, 3] - predicted[:3, 3]))
        if len(cameras) < 2 or score < 0.65 or innovation > reach:
            self.reason = 'rejected inconsistent masks or large pose jump; prediction only'
            self.support = len(cameras)
            return
        if not self.held_by:
            measured_velocity = (candidate[:3, 3] - self.pose[:3, 3]) / dt
            self.velocity = 0.65 * measured_velocity + 0.35 * self.velocity if self.velocity_known else measured_velocity
            self.velocity_known = True
        self.pose, self.step, self.state = (candidate, step, state)
        if self.held_by:
            self.attachment = np.linalg.inv(hand_pose(state, self.held_by)) @ candidate
        self.fit_score, self.innovation, self.support = (score, innovation, len(cameras))
        self.reason = 'visually updated shared rigid pose'

    def result(self, step, state, predict_actions=0):
        age = max(0, step - self.step)
        horizon = int(np.clip(predict_actions, 0, 8))
        current = self.predict(step, state)
        future = self.predict(step + horizon, state)
        xyz = transform(self.crown[None], current)[0]
        uncertainty = 0.005 + min(0.03, self.innovation) + age * 0.002
        valid = age <= 10 and self.support >= 2
        return dict(valid=valid, status='observed' if age == 0 else 'predicted', note=self.reason, action_step=step, observation_step=self.step, age_actions=age, supporting_views=self.support, fit_score=round(self.fit_score, 4), held_by=self.held_by, body_center_xyz=transform(np.array([[0, 0, self.size[2] / 2]]), current)[0].tolist(), body_size_m=self.size.tolist(), body_rotation_matrix=current[:3, :3].tolist(), handle_xyz=xyz.tolist(), handle_tangent_xyz=(current[:3, :3] @ self.tangent).tolist(), handle_diameter_m=2 * self.radius, handle_arch_xyz=transform(self.handle[::4], current).tolist(), velocity_m_per_action=self.velocity.tolist(), velocity_known=self.velocity_known, predicted_handle_xyz=transform(self.crown[None], future)[0].tolist(), predict_actions=horizon, uncertainty_m=round(uncertainty, 4))

def observe(primitives, obs):
    from martingale.hosts import robodojo_tools as tools
    state = tools._shown(obs)
    views = {c: tools._view(c, state) for c in shape.CAMERAS}
    masks, robots = ({}, {})
    for c, view in views.items():
        color = tools._camera(obs, c)['color']
        masks[c] = {}
        for part, prompt in [('whole', 'light blue basket'), ('handle', 'basket handle')]:
            res = primitives.sam3_client.segment(color, text_prompt=prompt, min_score=0.2)
            masks[c][part] = np.asarray(res.mask, bool) if res.found else np.zeros(color.shape[:2], bool)
        robots[c] = tools.sight.robot_pixels(view, state, own=tools._WRISTS.get(c))
    return (views, masks, robots, state)

def track_basket(primitives, predict_actions=0, held_by=None, reset=False):
    from martingale.hosts import robodojo_tools as tools
    if held_by not in (None, '', 'left', 'right'):
        return {'error': 'held_by must be left, right, empty string, or None'}
    obs = tools._obs(primitives)
    status = primitives.env.get_status()
    step = int(status.get('step', 0))
    pictured = max(0, step - int(status.get('pictures_behind') or 0))
    track = None if reset else getattr(primitives, '_basket_track', None)
    if reset:
        primitives._basket_track = None
    if track is None:
        seed = tools.locate(primitives, 'light blue basket')
        if not seed.get('center_xyz'):
            return dict(valid=False, note='Need an initial basket view; move to a clear view and retry.')
        views, masks, robots, state = observe(primitives, obs)
        x, y, _ = seed['center_xyz']
        yaw = np.deg2rad(seed['long_axis_deg'])
        initial = [x, y, yaw, seed['length_m'], seed['width_m'], seed['height_m'], 0.85]
        bounds = [(x - 0.15, x + 0.15), (y - 0.2, y + 0.2), (yaw - 0.6, yaw + 0.6), (0.12, 0.65), (0.06, 0.45), (0.03, 0.28), (0.6, 1.0)]
        body, bi = shape.fit_model(views, masks, robots, 'body', bounds, initial)
        if body is None or bi['mask_fit_score'] < 0.65:
            return dict(valid=False, note='Body fit needs two consistent views; retain clearance.')
        x, y, yaw, length, width, height, taper = body
        initial = [x, y, yaw + np.pi / 2, width / 2, shape.kin.TABLE_Z + height, 0.09, 0.005]
        bounds = [(x - 0.09, x + 0.09), (y - 0.09, y + 0.09), (yaw, yaw + np.pi), (0.04, 0.3), (shape.kin.TABLE_Z + height - 0.04, shape.kin.TABLE_Z + height + 0.025), (0.025, 0.2), (0.002, 0.015)]
        handle, hi = shape.fit_model(views, masks, robots, 'handle', bounds, initial)
        if handle is None or hi['mask_fit_score'] < 0.65:
            return dict(valid=False, note='Handle fit needs two consistent views; retain clearance.')
        track = BasketTrack(body, handle, pictured, state, min(bi['mask_fit_score'], hi['mask_fit_score']))
        primitives._basket_track = track
    if held_by is not None:
        track.set_holder(held_by)
    if pictured > track.seen_step:
        views, masks, robots, state = observe(primitives, obs)
        track.update(views, masks, robots, pictured, state)
    result = track.result(step, obs['state'], predict_actions)
    return result
track_basket._readonly = True

def after_action(primitives):
    """Refresh active tracks every five actions; never issue additional actions."""
    track = getattr(primitives, '_basket_track', None)
    if track is None:
        return
    status = primitives.env.get_status()
    pictured = max(0, int(status.get('step', 0)) - int(status.get('pictures_behind') or 0))
    if pictured - track.seen_step >= 5:
        try:
            track_basket(primitives)
        except Exception as exc:
            track.reason = f'tracker observation failed: {type(exc).__name__}: {exc}'
            track.support = 0
