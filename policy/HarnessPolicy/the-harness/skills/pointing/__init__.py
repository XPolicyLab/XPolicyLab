"""Open-vocabulary pointing and localization (pointer model, or the VLM)."""

import math


# localize_3d ranks estimates this close to a fingertip last: the pointer marks the
# fingers themselves, and a held object reaches further from them.
GRIPPER_NEAR_M, HELD_NEAR_M = 0.05, 0.10
# localize_3d: estimates farther than this behind the arm bases are implausible.
PLAUSIBLE_BEHIND_M = 0.15


class Pointing:
    def point_to(self, observation, prompt, camera="cam_head", tiles=None):
        """Pixels (u, v) in observation.images[camera] that match the prompt. With tiles
        (default: for cam_head, where objects look small), a prompt found nowhere in the
        whole image is searched again in enlarged tiles."""
        model = self.pointer or self._vision()
        image = observation.images[camera]
        points = model.point(image, prompt)
        if points or not (camera == "cam_head" if tiles is None else tiles):
            return points
        return self._point_in_tiles(model, image, prompt)

    def _point_in_tiles(self, model, image, prompt, grid=2, overlap=0.25, scale=2):
        """Point in overlapping tiles, each enlarged `scale` times, and map the points
        back to the full image (objects of a few pixels become visible to the model)."""
        import numpy as np
        from PIL import Image

        height, width = np.asarray(image).shape[:2]
        tile_w, tile_h = int(width / grid * (1 + overlap)), int(height / grid * (1 + overlap))
        found = []
        for row in range(grid):
            for col in range(grid):
                x0 = min(int(col * width / grid), width - tile_w)
                y0 = min(int(row * height / grid), height - tile_h)
                tile = Image.fromarray(np.asarray(image)[y0:y0 + tile_h, x0:x0 + tile_w])
                tile = np.asarray(tile.resize((tile_w * scale, tile_h * scale), Image.BICUBIC))
                for u, v in model.point(tile, prompt):
                    point = (x0 + u / scale, y0 + v / scale)
                    if all(math.dist(point, other) > 8 for other in found):
                        found.append(point)
        return found

    def localize(self, observation, prompt, camera="cam_head", z=None):
        """Point at `prompt` in one camera and intersect the pixel ray with the
        horizontal plane at height z (default: the table top). Returns
        {"pixel": (u, v), "position": [x, y, z]} for each match, best first.
        For tall objects pass the height of the point you expect the model to pick,
        or triangulate the same point from two cameras."""
        from harness_policy.geometry import pixel_to_plane

        height = self.table_z if z is None else float(z)
        results = []
        for u, v in self.point_to(observation, prompt, camera):
            position = pixel_to_plane(observation, camera, u, v, height)
            if position is not None:
                results.append({"pixel": (u, v), "position": position.tolist()})
        return results

    def localize_3d(self, observation, prompt, cameras=("cam_head", "cam_left_wrist", "cam_right_wrist"),
                    max_residual=0.03):
        """3D position of a single object named by `prompt`, triangulated from the
        cameras that see it (point_to in each). Unlike localize, this does not assume
        a height. Returns {"position", "cameras", "residual_m", "single_view"}. When several
        objects match the prompt, points are paired across views by geometric
        consistency (head camera first). When no two views agree (e.g. an arm occludes
        the object from the head camera), the position is one camera's ray at 2 cm above
        the table and single_view is True. None when no camera sees it."""
        import numpy as np

        from harness_policy.geometry import pixel_ray, pixel_to_plane, triangulate

        candidates = {camera: self.point_to(observation, prompt, camera) for camera in cameras}
        candidates = {camera: points for camera, points in candidates.items() if points}

        def residuals(position, pixels):
            result = {}
            for camera, (u, v) in pixels.items():
                origin, direction = pixel_ray(observation, camera, u, v)
                offset = position - origin
                result[camera] = float(np.linalg.norm(offset - direction * float(offset @ direction)))
            return result

        bases = ((getattr(self, "profile", None) or {}).get("arm_bases") or {}).values()
        behind = min((float(base[1]) for base in bases), default=None)

        def plausible(position):
            # On or above the table, and not behind the arm bases (where a wrong pairing
            # of pixels across cameras can land an estimate).
            return (self.table_z - 0.02 <= float(position[2]) <= self.table_z + 0.6
                    and (behind is None or float(position[1]) >= behind - PLAUSIBLE_BEHIND_M))

        # The pointer sometimes marks the robot's own fingers; such estimates rank last
        # (a held object is still found when nothing else matches).
        state = getattr(self, "state", None) or {}
        tips = [(np.asarray(self.fingertip_of(state[f"{side}_ee_pose"])),
                 HELD_NEAR_M if getattr(self, "groups", None) and self._holding_known(f"{side}_arm") else GRIPPER_NEAR_M)
                for side in ("left", "right") if state.get(f"{side}_ee_pose")]

        def at_gripper(position):
            return any(np.linalg.norm(position - tip) < near for tip, near in tips)

        # Several instances may match the prompt and each view may point at a different
        # one: pair points across views by geometric consistency, not by order.
        order = sorted(candidates, key=lambda camera: camera != "cam_head")
        best = None
        for first_index, first in enumerate(order):
            for rank, pixel in enumerate(candidates[first]):
                for second in order[first_index + 1:]:
                    for other in candidates[second]:
                        pixels = {first: pixel, second: other}
                        position = triangulate(observation, pixels)
                        error = max(residuals(position, pixels).values())
                        if error > max_residual or not plausible(position):
                            continue
                        # A third view that agrees joins the estimate.
                        for third in order:
                            if third in pixels:
                                continue
                            for extra in candidates[third]:
                                trial = {**pixels, third: extra}
                                refined = triangulate(observation, trial)
                                if max(residuals(refined, trial).values()) <= max_residual and plausible(refined):
                                    pixels, position = trial, refined
                                    break
                        key = (at_gripper(position), first_index, rank, -len(pixels), error)
                        if best is None or key < best[0]:
                            best = (key, position, pixels)
        def ray_at_gripper(camera, u, v):
            # A one-view estimate is a ray: when it passes by the fingertips, the pixel
            # shows the gripper or what it holds, wherever the ray meets the table.
            origin, direction = pixel_ray(observation, camera, u, v)
            for tip, near in tips:
                offset = tip - origin
                if np.linalg.norm(offset - direction * float(offset @ direction)) < near:
                    return True
            return False

        planar = []
        for camera in order:
            for u, v in candidates[camera]:
                position = pixel_to_plane(observation, camera, u, v, self.table_z + 0.02)
                if position is not None:
                    planar.append((ray_at_gripper(camera, u, v), len(planar), position, camera))
        # Anything away from the fingers first; then two views over one.
        near_best = best is not None and best[0][0]
        away = [item for item in planar if not item[0]]
        if best is not None and (not near_best or not away):
            _key, position, pixels = best
            errors = residuals(position, pixels)
            return {"position": position.tolist(), "cameras": sorted(pixels),
                    "residual_m": sum(errors.values()) / len(errors), "single_view": False}
        if planar:
            _near, _index, position, camera = min(planar, key=lambda item: item[:2])
            return {"position": position.tolist(), "cameras": [camera], "residual_m": None,
                    "single_view": True}
        return None
