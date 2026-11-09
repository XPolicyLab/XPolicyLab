"""Frames, poses and camera geometry: no model and no motion."""

from harness_policy.geometry import fingertip_of, link6_for_tip, matrix_quaternion, pose_matrix
from harness_policy.runtime.session import CAMERAS


class Geometry:
    def fingertip_of(self, pose):
        """Fingertip position [x, y, z] of a link6 pose [x, y, z, qw, qx, qy, qz]."""
        return fingertip_of(pose, self.tip_offset)

    def link6_for_tip(self, tip, orientation):
        """link6 pose [x, y, z, qw, qx, qy, qz] placing the fingertips at `tip`."""
        return link6_for_tip(tip, orientation, self.tip_offset)

    def pixel_to_plane(self, observation, camera, u, v, z=None):
        """Where the ray through pixel (u, v) meets the plane at height z (default: the table top)."""
        from harness_policy.geometry import pixel_to_plane
        return pixel_to_plane(observation, camera, u, v, self.table_z if z is None else z)

    def mask_to_plane(self, observation, camera, mask, z=None, step=4):
        """World points where the rays through a mask meet the plane at height z (default: the table top)."""
        from harness_policy.geometry import mask_to_plane
        return mask_to_plane(observation, camera, mask, self.table_z if z is None else z, step)

    def transform(self, pose, from_frame, to_frame):
        """Express a point [x, y, z] or pose [x, y, z, qw, qx, qy, qz] given in from_frame
        in to_frame. Frames: "world", "left_arm_base", "right_arm_base", "left_link6",
        "right_link6", and the camera names (OpenCV convention: z forward, y down)."""
        import numpy as np

        values = [float(v) for v in pose]
        if len(values) not in (3, 7):
            raise ValueError("pose is [x, y, z] or [x, y, z, qw, qx, qy, qz]")
        cameras = [f for f in (from_frame, to_frame) if f in CAMERAS]
        observation = self.get_observation(cameras=cameras) if cameras else None
        world = self._frame(to_frame, observation)
        relative = np.linalg.inv(world) @ self._frame(from_frame, observation)
        matrix = pose_matrix(values if len(values) == 7 else values + [1.0, 0.0, 0.0, 0.0])
        result = relative @ matrix
        out = [float(v) for v in result[:3, 3]]
        return out if len(values) == 3 else out + matrix_quaternion(result[:3, :3])

    def _frame(self, name, observation):
        """4x4 transform from `name` to the world (env) frame."""
        import numpy as np

        if name == "world":
            return np.eye(4)
        if name in ("left_arm_base", "right_arm_base"):
            base = [float(v) for v in self.profile["arm_bases"][name.replace("_base", "")]]
            return pose_matrix(base if len(base) == 7 else base[:3] + [1.0, 0.0, 0.0, 0.0])
        if name in ("left_link6", "right_link6"):
            return pose_matrix(self.get_state()[f"{name.split('_')[0]}_ee_pose"])
        if name in CAMERAS:
            return np.asarray(observation.camera_to_world[name], dtype=float)
        raise ValueError(f"Unknown frame {name!r}; frames are world, left_arm_base, right_arm_base, "
                         f"left_link6, right_link6, {', '.join(CAMERAS)}")


def exports(session):
    from harness_policy import geometry

    # Camera geometry uses the OpenCV convention; world = env frame.
    return {"transform": session.transform, "mask_to_plane": session.mask_to_plane,
            "down_with_yaw": geometry.down_with_yaw, "project": geometry.project, "pixel_ray": geometry.pixel_ray,
            "pixel_to_plane": session.pixel_to_plane, "triangulate": geometry.triangulate,
            "TABLE_Z": session.table_z, "FORWARD": list(geometry.FORWARD), "DOWN": list(geometry.DOWN),
            "link6_for_tip": session.link6_for_tip, "fingertip_of": session.fingertip_of}
