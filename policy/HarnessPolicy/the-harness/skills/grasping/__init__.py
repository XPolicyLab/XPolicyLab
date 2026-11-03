"""Grasp planning on a mask: top-down, pitched, side, part and rim grasps."""

import math

from harness_policy.geometry import down_with_yaw, matrix_quaternion


# plan_grasp calls an object graspable when it is this much narrower than the open
# gripper (profile gripper_open_width_m; ARX X5 default: 2 x 0.044 m finger travel).
GRASP_WIDTH_MARGIN_M, GRIPPER_OPEN_WIDTH_M = 0.008, 0.088
# Approach pitches (degrees down from horizontal) plan_grasp tries after top-down.
GRASP_PITCHES_DEG = (60, 45, 30, 0)
# plan_grasp, last resort for a footprint too wide and without a thin part (a bowl
# seen from above is a disc): pinch the wall at the edge nearest the arm, the fingers
# centered this far inside the edge, this far below the outline (the rim's height).
RIM_INSET_M, RIM_BELOW_M = 0.012, 0.02
# plan_grasp: the strip along the closing line through the footprint's center must
# hold at least this fraction of the points a uniform footprint puts there (else the
# center is a gap, and a part is grasped instead).
MATERIAL_MIN_FRACTION = 0.3


class Grasping:
    def plan_grasp(self, observation, mask, camera="cam_head", z=None, lift=0.10, top_down=False, arm=None,
                   prefer_side=False, grasp_z=None):
        """Grasp of the object in `mask` (from `camera`).

        Projects the mask onto the plane at height z (default: 2 cm above the table)
        and plans at the footprint center. Top-down: the fingers close across the
        footprint's short axis. Angled: link6 approaches from the arm's base toward the
        object, tilted down by a pitch (60, 45 or 30 degrees, or 0 for a horizontal
        side grasp), and the fingers close horizontally across it. Returns tip position,
        approach ("top_down", "pitched_60", "pitched_45", "pitched_30" or "side"), closing
        direction, width across the fingers (m), link6 grasp pose and a pregrasp pose
        `lift` before it (above it for top-down, back along the approach otherwise),
        graspable (width below the open gripper), and, without top_down, the arm and
        whether both poses have IK solutions (reachable).

        top_down=True returns the top-down grasp. Otherwise the first candidate in the
        order top-down, pitched 60, 45, 30, side (with prefer_side, for tall objects:
        side, pitched 30, 45, 60, top-down) that is graspable and reachable for `arm`
        (default: the arm whose base is nearer in x). When the whole object is too wide,
        a part grasp ("top_down_part"): top-down across a thin part of the footprint (a
        ring's band, a rim, a handle) whose finger landing spots are clear of the rest
        of the object; also when the closing line through the center crosses little of
        the object (a ring's or a crown's hole, the gap of a bent tool). Else the
        top-down one. Pass z at the height of the mask's outline (_measure estimates it
        from two wrist views): projecting a tall object's mask onto a plane far below
        its outline inflates the footprint, most for close wrist cameras. The fingers
        close at grasp_z (default z)."""
        from harness_policy.geometry import mask_to_plane

        height = self.table_z + 0.02 if z is None else float(z)
        points = mask_to_plane(observation, camera, mask, height)
        if len(points) < 10:
            return None
        outline = height
        if grasp_z is not None:
            height = float(grasp_z)
            points[:, 2] = height
        top = self._top_down_grasp(points, height, float(lift))
        if top_down:
            return top
        center = points.mean(axis=0)
        if arm is None:
            bases = self.profile["arm_bases"]
            arm = min(("left_arm", "right_arm"), key=lambda name: abs(bases[name][0] - center[0]))
        self._arm(arm)
        candidates = [top] + [self._angled_grasp(points, height, float(lift), arm, pitch)
                              for pitch in GRASP_PITCHES_DEG]
        if prefer_side:
            candidates.reverse()
        for candidate in candidates:
            candidate["arm"] = arm
            candidate["reachable"] = None
        solid = self._solid_center(points)
        for candidate in candidates if solid else []:
            if not candidate["graspable"]:
                continue
            candidate["reachable"] = (self.ik(arm, candidate["grasp"]) is not None
                                      and self.ik(arm, candidate["pregrasp"]) is not None)
            if candidate["reachable"]:
                return candidate
        if not solid or not any(candidate["graspable"] for candidate in candidates):
            base = self.profile["arm_bases"][arm][:2]
            for part in self._part_grasps(points, height, float(lift), base):
                part["arm"] = arm
                part["reachable"] = (self.ik(arm, part["grasp"]) is not None
                                     and self.ik(arm, part["pregrasp"]) is not None)
                if part["reachable"]:
                    return part
            rim = self._rim_pinch(points, max(height, outline - RIM_BELOW_M), float(lift), base)
            rim["arm"] = arm
            rim["reachable"] = (self.ik(arm, rim["grasp"]) is not None and self.ik(arm, rim["pregrasp"]) is not None)
            if rim["reachable"]:
                return rim
        return top

    def _rim_pinch(self, points, height, lift, base_xy):
        """Top-down pinch across the wall at the footprint's edge nearest the arm base
        (a bowl's or a cup's rim): the fingers close radially, one inside the object and
        one outside, RIM_INSET_M inside the edge."""
        import numpy as np

        xy = points[:, :2]
        center = xy.mean(axis=0)
        edge = xy[int(np.argmin(np.linalg.norm(xy - np.asarray(base_xy, dtype=float), axis=1)))]
        radial = edge - center
        radial = radial / max(float(np.linalg.norm(radial)), 1e-9)
        spot = edge - radial * RIM_INSET_M
        yaw = math.atan2(radial[1], radial[0])
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        tip = [float(spot[0]), float(spot[1]), float(height)]
        grasp = self.link6_for_tip(tip, down_with_yaw(yaw))
        pregrasp = list(grasp)
        pregrasp[2] += lift
        return {"tip": tip, "approach": "top_down_rim", "closing_direction": [math.cos(yaw), math.sin(yaw), 0.0],
                "yaw": yaw, "width_m": 2 * RIM_INSET_M, "length_m": None, "grasp": grasp, "pregrasp": pregrasp,
                "graspable": True}

    @staticmethod
    def _solid_center(points, band=0.01):
        """Whether fingers closing across the footprint's center meet the object: the
        strip within `band` of the closing line holds at least MATERIAL_MIN_FRACTION of
        the points a uniform footprint puts there. Not so for a footprint whose center
        is a gap (two ends of a bent tool, the corners of a crown seen from above)."""
        import numpy as np

        xy = points[:, :2] - points[:, :2].mean(axis=0)
        _values, vectors = np.linalg.eigh(np.cov(xy.T))
        along = xy @ vectors[:, 1]
        length = float(np.ptp(along))
        if length <= 4 * band:
            return True
        expected = len(xy) * (2 * band) / length
        return int(np.sum(np.abs(along) < band)) >= max(3.0, MATERIAL_MIN_FRACTION * expected)

    def _part_grasps(self, points, height, lift, base_xy, radius=0.03, band=0.015, finger=0.015, limit=5):
        """Top-down grasps across thin parts of a footprint too wide to grasp whole:
        around each sampled point, the local principal axes give the part's direction
        and its thickness across; the fingers close across it, and their landing spots
        (half the opening plus a finger to each side) must hold no other part. Best
        first: the most substantial part (most footprint within `radius`, not a thin
        tip), then the one nearer the arm base."""
        import numpy as np

        xy = points[:, :2]
        opening = self._graspable_width() + GRASP_WIDTH_MARGIN_M
        step = max(1, len(xy) // 60)
        found = []
        for center in xy[::step]:
            near = xy[np.linalg.norm(xy - center, axis=1) < radius]
            if len(near) < 5:
                continue
            _values, vectors = np.linalg.eigh(np.cov((near - near.mean(axis=0)).T))
            across = vectors[:, 0]
            offsets = (near - center) @ across
            thickness = float(np.ptp(offsets))
            if thickness > opening - 0.02:
                continue
            middle = center + across * float(offsets.min() + offsets.max()) / 2
            # The closing stroke must be clear except for the part itself.
            along = (xy - middle) @ across
            beside = np.abs((xy - middle) @ np.array([-across[1], across[0]]))
            in_stroke = (np.abs(along) < opening / 2 + finger) & (beside < band)
            outside_part = np.abs(along) > thickness / 2 + 0.005
            if np.any(in_stroke & outside_part):
                continue
            distance = float(np.linalg.norm(middle - np.asarray(base_xy, dtype=float)))
            found.append((-len(near), round(distance, 2), thickness, float(middle[0]), float(middle[1]),
                          float(math.atan2(across[1], across[0]))))
        found.sort()
        grasps = []
        for _mass, _distance, thickness, x, y, yaw in found[:limit]:
            if yaw > math.pi / 2:
                yaw -= math.pi
            elif yaw < -math.pi / 2:
                yaw += math.pi
            tip = [x, y, height]
            grasp = self.link6_for_tip(tip, down_with_yaw(yaw))
            pregrasp = list(grasp)
            pregrasp[2] += lift
            grasps.append({"tip": tip, "approach": "top_down_part", "closing_direction": [math.cos(yaw), math.sin(yaw), 0.0],
                           "yaw": yaw, "width_m": thickness, "length_m": None, "grasp": grasp, "pregrasp": pregrasp,
                           "graspable": True})
        return grasps

    def _top_down_grasp(self, points, height, lift):
        import numpy as np

        center = points.mean(axis=0)
        centered = points[:, :2] - center[:2]
        values, vectors = np.linalg.eigh(np.cov(centered.T))
        major, minor = vectors[:, 1], vectors[:, 0]
        width = float(np.ptp(centered @ minor))
        length = float(np.ptp(centered @ major))
        yaw = math.atan2(minor[1], minor[0])
        # Fingers close along (cos yaw, sin yaw); keep yaw within +-90 deg of the start.
        if yaw > math.pi / 2:
            yaw -= math.pi
        elif yaw < -math.pi / 2:
            yaw += math.pi
        orientation = down_with_yaw(yaw)
        tip = [float(center[0]), float(center[1]), height]
        grasp = self.link6_for_tip(tip, orientation)
        pregrasp = list(grasp)
        pregrasp[2] += lift
        return {"tip": tip, "approach": "top_down", "closing_direction": [math.cos(yaw), math.sin(yaw), 0.0],
                "yaw": yaw, "width_m": width, "length_m": length, "grasp": grasp, "pregrasp": pregrasp,
                # Fully open, the fingers are about 0.088 m apart (2 x 0.044 m travel).
                "graspable": width < self._graspable_width()}

    def _angled_grasp(self, points, height, lift, arm, pitch_deg):
        """link6 +x (the approach axis) points from the arm's base toward the object,
        tilted down by pitch_deg from horizontal; the fingers (link6 local y) close
        horizontally across the approach. Pitch 0 is FORWARD turned toward the object."""
        import numpy as np

        center = points.mean(axis=0)
        base = self.profile["arm_bases"][arm]
        heading = math.atan2(center[1] - base[1], center[0] - base[0])
        pitch = math.radians(pitch_deg)
        horizontal = np.array([math.cos(heading), math.sin(heading)])
        approach = np.array([math.cos(pitch) * horizontal[0], math.cos(pitch) * horizontal[1], -math.sin(pitch)])
        closing = np.array([-math.sin(heading), math.cos(heading), 0.0])
        rotation = np.column_stack([approach, closing, np.cross(approach, closing)])
        orientation = matrix_quaternion(rotation.tolist())
        centered = points[:, :2] - center[:2]
        width = float(np.ptp(centered @ closing[:2]))
        depth = float(np.ptp(centered @ horizontal))
        tip = [float(center[0]), float(center[1]), height]
        grasp = self.link6_for_tip(tip, orientation)
        pregrasp = [grasp[i] - lift * approach[i] for i in range(3)] + list(grasp[3:])
        return {"tip": tip, "approach": f"pitched_{pitch_deg}" if pitch_deg else "side",
                "closing_direction": [float(v) for v in closing], "yaw": heading, "pitch_deg": pitch_deg,
                "width_m": width, "length_m": depth, "grasp": grasp, "pregrasp": pregrasp,
                "graspable": width < self._graspable_width()}

    def _graspable_width(self):
        profile = getattr(self, "profile", None) or {}
        return float(profile.get("gripper_open_width_m", GRIPPER_OPEN_WIDTH_M)) - GRASP_WIDTH_MARGIN_M
