"""RGB target feature motion, not contact/force or evaluator success feedback.

Features originate ONLY in the language-selected RGB instance. Occlusion,
insufficient parallax and inconsistent motion return unknown, never attached.
The ee_joint_state observation is a previous limited command, NOT a sensor.
"""
import cv2
import numpy as np
from .geometry import Camera


class TargetMotion:
    def __init__(self, calibration):
        self.camera = Camera(calibration['head'])
        self.reset()

    def reset(self):
        self.gray = self.points = self.world = self.start = None
        self.anchor_tcp = None
        self.rgb = self.reference_rgb = self.support = None
        self.surface_z = None
        self.evidence = {'status': 'unknown', 'reason': 'not_initialized'}

    def initialize(self, rgb, support, surface_z):
        self.reset()
        if support is None or np.shape(support) != rgb.shape[:2]:
            return
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.rgb = self.reference_rgb = rgb.copy()
        self.support = cv2.erode(np.asarray(support, np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        self.surface_z = float(surface_z)
        mask = cv2.erode(np.asarray(support, np.uint8), np.ones((3, 3), np.uint8)) * 255
        points = cv2.goodFeaturesToTrack(gray, 60, .015, 3, mask=mask, blockSize=3)
        self.gray = gray
        if points is None or len(points) < 6:
            self.evidence = {'status': 'unknown', 'reason': 'insufficient_instance_texture'}
            return
        self.points = points.reshape(-1, 2)
        self.world = np.array([self.camera.pixel_to_plane(p, surface_z) for p in self.points])
        self.evidence = {'status': 'unknown', 'reason': 'awaiting_test_lift', 'features': len(points)}

    def update(self, rgb):
        self.rgb = rgb.copy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.points is None or len(self.points) < 6:
            self.gray = gray
            self.evidence = {'status': 'unknown', 'reason': 'target_occluded_or_lost'}
            return
        old = self.points.astype(np.float32).reshape(-1, 1, 2)
        new, status, err = cv2.calcOpticalFlowPyrLK(self.gray, gray, old, None,
                                                  winSize=(15, 15), maxLevel=2)
        if new is None:
            self.points = None
            self.gray = gray
            self.evidence = {'status': 'unknown', 'reason': 'flow_failed'}
            return
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, new, None,
                                                      winSize=(15, 15), maxLevel=2)
        if back is None:
            self.points = None
            self.gray = gray
            self.evidence = {'status': 'unknown', 'reason': 'backward_flow_failed'}
            return
        p = new.reshape(-1, 2)
        keep = ((status[:, 0] != 0) & (back_status[:, 0] != 0) & (err[:, 0] < 18)
                & (np.linalg.norm(back.reshape(-1, 2)-old.reshape(-1, 2), axis=1) < .8)
                & np.isfinite(p).all(axis=1) & (p[:, 0] >= 2) & (p[:, 0] < gray.shape[1]-2)
                & (p[:, 1] >= 2) & (p[:, 1] < gray.shape[0]-2))
        self.points, self.world = p[keep], self.world[keep]
        if self.start is not None:
            self.start = self.start[keep]
        self.gray = gray

    def anchor(self, tcp):
        """Called before closing. Never reseed on an occluding gripper."""
        self.anchor_tcp = np.asarray(tcp, float).copy()
        if self.points is None or len(self.points) < 6:
            self.start = None
            return
        self.start = self.points.copy()
        # Reference height is an RGB estimate, not simulator depth.
        height = float(np.median(self.world[:, 2]))
        self.world = np.array([self.camera.pixel_to_plane(p, height) for p in self.start])

    def stationary_appearance(self, tcp):
        """Reacquire static appearance after occlusion without switching identity.

Only a strong match on the ORIGINAL instance material, a poor match at the
rigid-attachment prediction, and enough geometric image motion vote empty.
This never votes following. Occlusion/low contrast/moved targets stay unknown.
"""
        if self.anchor_tcp is None or self.reference_rgb is None or self.support is None:
            return None
        delta = np.asarray(tcp)-self.anchor_tcp
        if delta[2] < .025: return None
        yy, xx = np.nonzero(self.support)
        if len(xx) < 60: return None
        stride = max(1, len(xx)//500)
        yy, xx = yy[::stride], xx[::stride]
        uv = np.c_[xx, yy]
        xyz = np.array([self.camera.pixel_to_plane(p, self.surface_z) for p in uv])
        predicted = self.camera.project(xyz+delta)
        if np.median(np.linalg.norm(predicted-uv, axis=1)) < 6: return None
        moved = np.rint(predicted).astype(int)
        if not ((moved >= 0).all() and (moved[:, 0] < self.rgb.shape[1]).all()
                and (moved[:, 1] < self.rgb.shape[0]).all()): return None
        template = self.reference_rgb[yy, xx].astype(float)
        # Flat tabletop-coloured or uninformative regions cannot certify empty.
        ring = cv2.dilate(self.support.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool) & ~self.support
        background = np.median(self.reference_rgb[ring], axis=0) if ring.any() else np.median(template, axis=0)
        if np.linalg.norm(np.median(template, axis=0)-background) < 35: return None
        static_error = np.linalg.norm(self.rgb[yy, xx].astype(float)-template, axis=1)
        moved_error = np.linalg.norm(self.rgb[moved[:, 1], moved[:, 0]].astype(float)-template, axis=1)
        visible = float(np.mean(static_error < 35))
        a, b = float(np.median(static_error)), float(np.median(moved_error))
        if visible >= .8 and a < 20 and b > a+25:
            return dict(status='stationary', reason='original_target_material_still_on_table',
                        source='head_rgb_static_instance_reacquisition', is_task_success=False,
                        visible_match_fraction=visible, stationary_rgb_error=a, attached_rgb_error=b)
        return None

    def assess(self, tcp):
        result = {'status': 'unknown', 'reason': 'insufficient_visible_target',
                  'source': 'head_rgb_instance_flow', 'is_task_success': False}
        if self.start is None or self.points is None or len(self.points) < 6:
            result = self.stationary_appearance(tcp) or result
            self.evidence = result
            return result
        delta = np.asarray(tcp)-self.anchor_tcp
        predicted = self.camera.project(self.world+delta)
        expected = predicted-self.start
        motion = self.points-self.start
        signal = float(np.median(np.linalg.norm(expected, axis=1)))
        stationary = float(np.median(np.linalg.norm(motion, axis=1)))
        attached = float(np.median(np.linalg.norm(self.points-predicted, axis=1)))
        spread = float(np.linalg.norm(np.ptp(self.start, axis=0)))
        result.update(features=len(self.points), expected_motion_px=signal,
                      stationary_residual_px=stationary, attached_residual_px=attached,
                      tcp_rise_m=float(delta[2]), reference_spread_px=spread)
        if delta[2] >= .025 and signal >= 5 and spread >= 8:
            votes = np.linalg.norm(self.points-predicted, axis=1) < 4
            if attached < 3 and stationary > 4 and votes.mean() >= .75:
                result.update(status='following', reason='target_tracks_test_lift')
            elif stationary < 2.5 and attached > 5:
                result.update(status='stationary', reason='visible_target_did_not_follow')
        if result['status'] == 'unknown':
            result = self.stationary_appearance(tcp) or result
        self.evidence = result
        return result
