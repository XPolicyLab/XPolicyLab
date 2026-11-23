# Copyright © 2026 Martingale. All Rights Reserved.
import hashlib
from pathlib import Path
import time
import uuid

import numpy as np

from .temporal_geometry import MaskView, measure_static_pair, measure_planar_pair


class GeometryHistory:
    """At most four immutable views; expired handles fail rather than reuse data."""
    def __init__(self, archive):
        self.archive = Path(archive)
        self.episode = uuid.uuid4().hex
        self.frames = {}
        self.prompts = {}

    def capture(self, robo, text_prompt, camera="cam_head"):
        """Segment CURRENT RGB and retain its calibration/robot mask.

        Returns an opaque handle, step, mask score/hash and limitations. Full mask
        evidence is archived once per explicit call, not on physics steps. No
        motion, object identity proof or success feedback is used.
        """
        if camera not in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            raise ValueError("Unknown camera")
        if not isinstance(text_prompt, str) or not 1 <= len(text_prompt.strip()) <= 256:
            raise ValueError("Expected short object description")
        from martingale.hosts import robodojo_tools as tools, robodojo_sight as sight
        kit = robo._backend._toolkit
        record = robo._ledger.open("geometry_capture", dict(text_prompt=text_prompt, camera=camera), stateful=False)
        started = time.perf_counter()
        try:
            with kit._lock:
                primitives = kit._primitives
                obs = tools._obs(primitives)
                state = tools._shown(obs)
                view = tools._view(camera, state)
                if view is None:
                    raise ValueError("Missing camera pose")
                rgb = obs["vision"][camera]["color"]
                result = primitives.sam3_client.segment(rgb, text_prompt=text_prompt, min_score=.2)
                if not result.found:
                    out = dict(status="not_found", handle=None, episode_actions_taken=0)
                else:
                    mask = np.asarray(result.mask, dtype=bool)
                    own = {"cam_left_wrist": "left", "cam_right_wrist": "right"}.get(camera)
                    robot = sight.robot_pixels(view, state, own=own)
                    intrinsic = np.array([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]])
                    step = int(robo.state().step)
                    frame = MaskView(self.episode, step, camera, intrinsic, view.pose, mask, robot)
                    handle = uuid.uuid4().hex
                    self.archive.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(self.archive / (handle + ".npz"),
                                        mask=frame.mask, robot_mask=frame.robot_mask,
                                        intrinsic=frame.intrinsic, camera_to_world=frame.camera_to_world)
                    self.frames[handle] = frame
                    self.prompts[handle] = " ".join(text_prompt.lower().split())
                    while len(self.frames) > 4:
                        oldest = next(iter(self.frames))
                        del self.frames[oldest]
                        del self.prompts[oldest]
                    out = dict(status="captured", handle=handle, step=step, camera=camera,
                               score=float(result.score), mask_pixels=int(mask.sum()),
                               mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                               episode_actions_taken=0, stationarity_verified=False)
                record.ok = True
                record.summary = out
                return out
        except BaseException as error:
            record.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            record.elapsed_s = time.perf_counter() - started
            record.env_steps = 0

    def measure(self, first, second, *, stationarity_basis, planar_table=False):
        """Fit stored views of the same described object; never certify stability.

        Handles belong only to this episode and expire after four captures.
        Invalid/missing handles or different descriptions are rejected. A review
        string describes an assumption, not a stationarity oracle.
        planar_table=True fits a flat marking on the existing calibrated
        table plane. It is invalid for raised objects; consistency is not accuracy.
        """
        if not isinstance(first, str) or not isinstance(second, str):
            raise ValueError("Expected capture handles")
        if first not in self.frames or second not in self.frames:
            raise ValueError("Unknown or expired view handle")
        if self.prompts[first] != self.prompts[second]:
            raise ValueError("Different object descriptions")
        if stationarity_basis is not None and (not isinstance(stationarity_basis, str) or len(stationarity_basis) > 2000):
            raise ValueError("Expected bounded stationarity review")
        if not isinstance(planar_table, bool):
            raise ValueError("planar_table must be boolean")
        if planar_table:
            from martingale.hosts import robodojo_sight as sight
            return measure_planar_pair(self.frames[first], self.frames[second],
                                       plane_z=sight.kin.TABLE_Z, stationarity_basis=stationarity_basis)
        return measure_static_pair(self.frames[first], self.frames[second],
                                   stationarity_basis=stationarity_basis)
