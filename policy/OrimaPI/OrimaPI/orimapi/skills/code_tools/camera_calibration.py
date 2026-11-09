"""Read-only access to the public camera model already used by RGB geometry."""

import numpy as np

from orimapi.skills.code_skills.common.geometry import pose_matrix

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def rigid(value):
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (4, 4) or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Public camera calibration requires finite proper rigid transforms")
    return value.copy()


def describe(cameras, frames, robot_state, tip_offset):
    """Serialize calibrated frames only: no RGB interpretation, depth or action."""
    result = {}
    for name, frame in frames.items():
        if name not in CAMERAS or name not in cameras:
            raise ValueError("Unknown original camera")
        camera = cameras[name]
        if (
            type(camera.width) is not int or type(camera.height) is not int
            or camera.width <= 0 or camera.height <= 0
            or (frame["width"], frame["height"]) != (camera.width, camera.height)
        ):
            raise ValueError("Original image dimensions disagree with camera calibration")
        k = np.asarray(camera.k, dtype=float)
        if (
            k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0)
            or not np.isclose(k[1, 0], 0, atol=1e-8, rtol=0)
        ):
            raise ValueError("Invalid public pinhole intrinsics")
        world_from_optical = rigid(camera.optical)
        optical_from_world = rigid(camera.view)
        if not np.allclose(
            optical_from_world @ world_from_optical, np.eye(4), atol=1e-7, rtol=0
        ):
            raise ValueError("Public camera transforms are not mutual inverses")
        mount = {"type": "fixed_world_camera"}
        if name != "cam_high":
            side = "left" if name == "cam_left_wrist" else "right"
            value = robot_state.get(f"{side}_ee_pose")
            if value is None:
                raise ValueError("Current public wrist pose is unavailable")
            if (
                isinstance(tip_offset, bool) or tip_offset is None
                or not np.isfinite(tip_offset) or tip_offset < 0
            ):
                raise ValueError("Public wrist-to-TCP offset is unavailable")
            wrist_from_tcp = np.eye(4)
            wrist_from_tcp[0, 3] = float(tip_offset)
            world_from_tcp = rigid(pose_matrix(value) @ wrist_from_tcp)
            mount = {
                "type": "rigid_wrist_camera",
                "arm": side,
                "world_from_tcp": world_from_tcp.tolist(),
                "tcp_from_optical": (
                    np.linalg.inv(world_from_tcp) @ world_from_optical
                ).tolist(),
                "tcp_offset_from_wrist_m": float(tip_offset),
                "tcp_source": (
                    "Current public observed ee_pose wrist frame, translated by "
                    "the existing public gripper bias along wrist-local +X. "
                    "TCP is grasp center, not the observed wrist origin."
                ),
            }
        result[name] = {
            "frame_id": frame["frame_id"],
            "width": camera.width,
            "height": camera.height,
            "intrinsic_matrix_K": k.tolist(),
            "world_from_optical": world_from_optical.tolist(),
            "optical_from_world": optical_from_world.tolist(),
            "optical_axis_world": world_from_optical[:3, 2].tolist(),
            "camera_origin_world_m": world_from_optical[:3, 3].tolist(),
            "mount": mount,
        }
    return {
        "status": "public_calibration",
        "cameras": result,
        "coordinate_contract": (
            "Homogeneous column vectors; T_A_from_B maps B coordinates into A. "
            "World translations are metres. Optical axes: +X image-right, "
            "+Y image-down, +Z forward. Original pixel coordinates are (u,v). "
            "q_optical = optical_from_world @ [x_world,y_world,z_world,1]; "
            "[u,v,1] is proportional to K @ q_optical[:3], with q_optical[2]>0."
        ),
        "scope": (
            "The existing public pinhole/robot calibration at these CURRENT frames, "
            "not a new calibration estimate or observed object geometry. A pixel "
            "defines a ray, not object depth, contact, clearance or retention. "
            "For the same wrist's rigid mount, a future planned world_from_tcp "
            "can compose with tcp_from_optical to hypothesize its optical frame; "
            "this does not predict future RGB/occlusion, solve a viewing pose, "
            "authorize motion or replace existing geometry and safety gates."
        ),
        "official_success": None,
    }
