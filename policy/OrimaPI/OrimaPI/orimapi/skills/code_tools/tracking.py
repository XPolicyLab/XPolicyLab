"""Bounded per-episode RGB point tracking; missing evidence stays missing."""

from copy import deepcopy

import cv2
import numpy as np

from orimapi.skills.code_skills.common.geometry import rgb


class PointTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step = self.env_id = None
        self.frames, self.points = {}, {}
        self.revision = 0

    def update(self, observation):
        if self.env_id is not None and observation.env_id != self.env_id:
            raise ValueError("Point tracks belong to one environment; reset first")
        if self.step is not None and observation.step < self.step:
            raise ValueError("Point tracks cannot cross an episode reset")
        if observation.step == self.step:
            return
        current = {
            name: cv2.cvtColor(rgb(image), cv2.COLOR_RGB2GRAY)
            for name, image in observation.images.items()
        }
        gap = self.step is not None and observation.step != self.step + 1
        for name, previous in self.frames.items():
            rows = [
                p
                for p in self.points.values()
                if p["camera"] == name and p["status"] == "observed"
            ]
            if not rows:
                continue
            image = current.get(name)
            if gap or image is None or image.shape != previous.shape:
                for row in rows:
                    self._lost(row, "frame_gap_or_camera_change")
                continue
            before = np.asarray([p["pixel"] for p in rows], np.float32).reshape(
                -1, 1, 2
            )
            after, good, error = cv2.calcOpticalFlowPyrLK(
                previous, image, before, None, winSize=(21, 21), maxLevel=3
            )
            if after is None:
                for row in rows:
                    self._lost(row, "optical_flow_unavailable")
                continue
            back, reverse_good, _ = cv2.calcOpticalFlowPyrLK(
                image, previous, after, None, winSize=(21, 21), maxLevel=3
            )
            for index, row in enumerate(rows):
                pixel = after[index, 0]
                valid = (
                    back is not None
                    and good[index, 0]
                    and reverse_good[index, 0]
                    and np.isfinite(pixel).all()
                    and np.linalg.norm(back[index, 0] - before[index, 0]) <= 1.0
                    and float(error[index, 0]) <= 15.0
                    and 5 <= pixel[0] < image.shape[1] - 5
                    and 5 <= pixel[1] < image.shape[0] - 5
                )
                if valid:
                    old_patch = cv2.getRectSubPix(
                        previous, (11, 11), tuple(map(float, before[index, 0]))
                    )
                    new_patch = cv2.getRectSubPix(
                        image, (11, 11), tuple(map(float, pixel))
                    )
                    valid = float(old_patch.std()) >= 4 and float(new_patch.std()) >= 4
                    valid = (
                        valid
                        and np.mean(np.abs(old_patch.astype(float) - new_patch)) <= 18
                    )
                if not valid:
                    self._lost(row, "inconsistent_or_occluded_flow")
                    continue
                row.update(
                    pixel=pixel.astype(float).tolist(), last_seen_step=observation.step
                )
        self.frames = current
        self.env_id, self.step = observation.env_id, observation.step
        self.revision += 1

    @staticmethod
    def _lost(row, reason):
        row.update(status="lost", reason=reason, last_pixel=row.pop("pixel"))

    def mark(self, name, camera, pixel):
        if not isinstance(name, str) or not name.isidentifier() or len(name) > 64:
            raise ValueError("Point ID must be a bounded identifier")
        if name not in self.points and len(self.points) >= 64:
            raise ValueError("At most 64 tracked points per episode")
        if camera not in self.frames:
            raise ValueError("Selected camera has no current RGB")
        value = np.asarray(pixel, dtype=float)
        image = self.frames[camera]
        if (
            value.shape != (2,)
            or not np.isfinite(value).all()
            or not (
                5 <= value[0] < image.shape[1] - 5
                and 5 <= value[1] < image.shape[0] - 5
            )
        ):
            raise ValueError("Point must be inside current image with a 5-pixel margin")
        self.points[name] = dict(
            id=name,
            camera=camera,
            pixel=value.tolist(),
            status="observed",
            last_seen_step=self.step,
            provenance="current_rgb_selection",
            selection_revision=self.revision + 1,
        )
        self.revision += 1
        return deepcopy(self.points[name])

    def current(self, name):
        row = self.points.get(name)
        if (
            row is None
            or row["status"] != "observed"
            or row["last_seen_step"] != self.step
        ):
            raise ValueError(
                f"Point {name!r} is not currently observed; re-mark from fresh RGB"
            )
        return deepcopy(row)

    def describe(self):
        return deepcopy(list(self.points.values()))
