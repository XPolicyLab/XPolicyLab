import json
import torch


@torch.jit.script
def _sixd_rows_to_rotmat(x6d: torch.Tensor) -> torch.Tensor:
    """6D rotation (first two **rows** of rotation matrix) -> 3x3 matrix.

    Matches the convention of numpy ``sixd_to_rotmat`` in utils_validation.py
    and of ``euler_to_matrix_zyx_6d`` which outputs the first two rows.

    NOTE: ``rot6d_to_matrix`` in math_utils.py stacks vectors as **columns**
    (Zhou et al. convention), which is transposed w.r.t. what our data pipeline
    produces.  This helper avoids that mismatch.

    Args:
        x6d: (..., 6) — [row0(3), row1(3)]
    Returns:
        (..., 3, 3) rotation matrix
    """
    row0 = x6d[..., 0:3]
    row1 = x6d[..., 3:6]
    row2 = torch.cross(row0, row1, dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)  # stack as rows


@torch.jit.script
def _rotmat_to_euler_zyx(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> Euler angles [roll, pitch, yaw] (ZYX convention).

    Inverse of ``euler_to_matrix_zyx_6d_torch``.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    r00 = R[..., 0, 0]
    r10 = R[..., 1, 0]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    pitch = torch.atan2(-r20, torch.sqrt(r00 * r00 + r10 * r10))
    roll = torch.atan2(r21, r22)
    yaw = torch.atan2(r10, r00)

    return torch.stack((roll, pitch, yaw), dim=-1)


@torch.jit.script
def _relative2absolute(relative_actions, agent_pos):
    """
    Args:
        relative_actions: [B, T, C=20]
        agent_pos: [B, 1, C=20]
        [
            left_pos(3), left_rot6d(6), left_gripper(1),
            right_pos(3), right_rot6d(6), right_gripper(1)
        ]
    Returns:
        absolute_actions: [B, T, C'=14]
        [
            left_xyz(3), left_euler(3), left_grip(1),
            right_xyz(3), right_euler(3), right_grip(1)
        ]
    """

    B, T, C = relative_actions.shape
    assert C >= 20, f"Expected at least 20 action dims, got {C}"

    device = relative_actions.device
    dtype = relative_actions.dtype

    ref20 = agent_pos[:, 0, :20]  # [B, 20]
    rel = relative_actions[:, :, :20]

    rel_l_pos = rel[:, :, 0:3]  # [B, T, 3]
    rel_l_rot6 = rel[:, :, 3:9]  # [B, T, 6]
    rel_l_grip = rel[:, :, 9:10]  # [B, T, 1]

    rel_r_pos = rel[:, :, 10:13]  # [B, T, 3]
    rel_r_rot6 = rel[:, :, 13:19]  # [B, T, 6]
    rel_r_grip = rel[:, :, 19:20]  # [B, T, 1]

    ref_l_pos = ref20[:, 0:3].unsqueeze(1)  # [B, 1, 3]
    ref_l_rot6 = ref20[:, 3:9]  # [B, 6]
    ref_r_pos = ref20[:, 10:13].unsqueeze(1)  # [B, 1, 3]
    ref_r_rot6 = ref20[:, 13:19]  # [B, 6]

    # 6D -> rotation matrices (row-based, matching data pipeline convention)
    R_ref_l = _sixd_rows_to_rotmat(ref_l_rot6)  # [B, 3, 3]
    R_ref_r = _sixd_rows_to_rotmat(ref_r_rot6)  # [B, 3, 3]

    R_rel_l = _sixd_rows_to_rotmat(rel_l_rot6.reshape(B * T, 6)).reshape(B, T, 3, 3)
    R_rel_r = _sixd_rows_to_rotmat(rel_r_rot6.reshape(B * T, 6)).reshape(B, T, 3, 3)

    # R_abs = R_rel @ R_ref
    R_abs_l = torch.matmul(R_rel_l, R_ref_l.unsqueeze(1))  # [B, T, 3, 3]
    R_abs_r = torch.matmul(R_rel_r, R_ref_r.unsqueeze(1))  # [B, T, 3, 3]

    eul_l = _rotmat_to_euler_zyx(R_abs_l)  # [B, T, 3]
    eul_r = _rotmat_to_euler_zyx(R_abs_r)  # [B, T, 3]

    out = torch.zeros((B, T, 14), device=device, dtype=dtype)

    out[:, :, 0:3] = ref_l_pos + rel_l_pos
    out[:, :, 3:6] = eul_l
    out[:, :, 6:7] = rel_l_grip

    out[:, :, 7:10] = ref_r_pos + rel_r_pos
    out[:, :, 10:13] = eul_r
    out[:, :, 13:14] = rel_r_grip

    return out


@torch.jit.script
def _euler_xyz_to_rotmat(euler: torch.Tensor) -> torch.Tensor:
    """Euler angles (extrinsic xyz / intrinsic ZYX) → rotation matrix.

    Matches scipy ``Rotation.from_euler("xyz", [rx, ry, rz])``.
    R = Rz(rz) @ Ry(ry) @ Rx(rx).

    Args:
        euler: [..., 3]  (rx, ry, rz)
    Returns:
        [..., 3, 3] rotation matrices
    """
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    return torch.stack(
        [
            torch.stack(
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], dim=-1
            ),
            torch.stack(
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], dim=-1
            ),
            torch.stack([-sy, cy * sx, cy * cx], dim=-1),
        ],
        dim=-2,
    )


@torch.jit.script
def _apply_T_batch(T44: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 rigid transform to batched 3-D points.

    T44: [4, 4]   pts: [..., 3]   → [..., 3]
    """
    return torch.einsum("ij,...j->...i", T44[:3, :3], pts) + T44[:3, 3]


@torch.jit.script
def _pinhole_project(pts: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pinhole projection (no distortion).

    pts: [..., 3]   K: [3, 3]   → [..., 2]  (u, v)
    """
    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
    Z_safe = torch.where(Z.abs() < 1e-6, torch.ones_like(Z) * 1e-6, Z)
    u = K[0, 0] * X / Z_safe + K[0, 2]
    v = K[1, 1] * Y / Z_safe + K[1, 2]
    return torch.stack([u, v], dim=-1)


def intrinsics_roi_map(
    K_src: torch.Tensor,
    scale: float = 2.25,
    crop_x: float = 240.0,
    crop_y: float = 0.0,
    crop_mode: str = "pre",
) -> torch.Tensor:
    """ROI-style resize/crop of camera intrinsics, implemented in torch."""

    # Always operate in float64 for numerical stability
    K = K_src.to(dtype=torch.float64).clone()
    s = float(scale)
    cx, cy = float(K[0, 2].item()), float(K[1, 2].item())

    K[0, 0] /= s
    K[1, 1] /= s

    if crop_mode == "pre":
        K[0, 2] = (cx - float(crop_x)) / s
        K[1, 2] = (cy - float(crop_y)) / s
    elif crop_mode == "post":
        K[0, 2] = cx / s - float(crop_x)
        K[1, 2] = cy / s - float(crop_y)
    else:
        raise ValueError("crop_mode must be 'pre' or 'post'")
    return K


def resize_camera_intrinsics(
    K_src: torch.Tensor,
    src_wh,
    dst_wh,
) -> torch.Tensor:
    """Resize camera intrinsics from src_wh to dst_wh using pure torch."""

    # src_wh / dst_wh are tuples of ints; keep computation in float64
    scale = min(src_wh[0] / dst_wh[0], src_wh[1] / dst_wh[1])
    crop_x = (src_wh[0] - dst_wh[0] * scale) / 2.0
    crop_y = src_wh[1] - dst_wh[1] * scale
    r = intrinsics_roi_map(
        K_src,
        scale=scale,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_mode="pre",
    )
    return r


class ActionProjector2D:
    """Batched projection of absolute 14-dim actions to 2D pixel coordinates.

    Stores calibration matrices from a robot JSON (same format as sight_cone's
    ``read_cameras_for_bimanual_3views``).  Not an nn.Module – no learnable
    parameters; operates with ``torch.no_grad`` on detached tensors.

    Absolute action layout (same as ``_relative2absolute`` output):
        [0:3]  left_pos        (xyz in follow frame)
        [3:6]  left_euler      (rx, ry, rz – scipy "xyz" convention)
        [6]    left_gripper
        [7:10] right_pos
        [10:13] right_euler
        [13]   right_gripper
    """

    DEFAULT_GRIPPER_IN_EEF = (0.162, 0.0, 0.0)

    def __init__(
        self,
        calib_json_path: str,
        target_height: int = 480,
        target_width: int = 640,
        gripper_in_eef=None,
    ):

        if gripper_in_eef is None:
            gripper_in_eef = self.DEFAULT_GRIPPER_IN_EEF
        self.gripper_in_eef = torch.tensor(gripper_in_eef, dtype=torch.float64)
        self.target_height = target_height
        self.target_width = target_width

        with open(calib_json_path, "r") as f:
            calib = json.loads(f.read())
        intr = calib["calibration"]["intrinsics"]
        extr = calib["calibration"]["extrinsics"]["transform"]

        def _load_K(cam_key):
            K_np = intr[cam_key]["camera_matrix"]
            K_t = torch.tensor(K_np, dtype=torch.float64)
            return resize_camera_intrinsics(
                K_t,
                (intr[cam_key]["image_width"], intr[cam_key]["image_height"]),
                (target_width, target_height),
            )

        self.K_head = _load_K("head")

        M_g2cL = torch.tensor(extr["M_gripper_to_camL"], dtype=torch.float64)
        M_l2h = torch.tensor(extr["M_left_to_head"], dtype=torch.float64)
        self.M1_left = M_g2cL
        self.M2_left = torch.inverse(M_l2h)

        M_g2cR = torch.tensor(extr["M_gripper_to_camR"], dtype=torch.float64)
        M_r2h = torch.tensor(extr["M_right_to_head"], dtype=torch.float64)
        self.M1_right = M_g2cR
        self.M2_right = torch.inverse(M_r2h)

        self.K_left = _load_K("left")
        self.pre_transform_left = torch.inverse(M_g2cL) @ M_l2h
        self.extrinsics_left = M_g2cL.clone()

        self.K_right = _load_K("right")
        self.pre_transform_right = torch.inverse(M_g2cR) @ M_r2h
        self.extrinsics_right = M_g2cR.clone()

    # -- device migration ---------------------------------------------------

    _TENSOR_ATTRS = (
        "gripper_in_eef",
        "K_head",
        "M1_left",
        "M2_left",
        "M1_right",
        "M2_right",
        "K_left",
        "pre_transform_left",
        "extrinsics_left",
        "K_right",
        "pre_transform_right",
        "extrinsics_right",
    )

    def _apply(self, fn):
        """Follow parent module's device migration, but preserve dtype.

        Higher-level modules may call `.to(device=..., dtype=...)`, which is
        implemented via `_apply(fn)`. For geometric calibration we always keep
        tensors in float64 for numerical stability, and only follow `fn`'s
        **device** change.
        """
        # Infer target device by applying `fn` to a sample tensor, but ignore
        # any dtype change it requests.
        sample = None
        for name in self._TENSOR_ATTRS:
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor):
                sample = t
                break
        if sample is None:
            return self

        moved = fn(sample)
        target_device = moved.device

        for name in self._TENSOR_ATTRS:
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor):
                setattr(self, name, t.to(device=target_device, dtype=t.dtype))

        return self

    # -- projection ----------------------------------------------------------
    @torch.no_grad()
    def project(self, abs_actions: torch.Tensor, return_eef: int = 0) -> dict:
        """Project 3-D absolute actions to per-camera 2-D pixel coordinates.
        Args:
            abs_actions: [B, T, >=14]  (see class docstring for layout)
            return_eef:
                0 -> only return gripper
                1 -> only return eef
                2 -> return both gripper and eef
        """
        device = abs_actions.device
        # All geometric computation is done in float64 for numerical stability,
        # regardless of the dtype used by the caller or stored inside the
        # projector (those can be bfloat16/float16 after model.to()).
        act = abs_actions.detach().to(dtype=torch.float64, device=device)
        left_pos = act[..., 0:3]
        left_euler = act[..., 3:6]
        right_pos = act[..., 7:10]
        right_euler = act[..., 10:13]
        R_left = _euler_xyz_to_rotmat(left_euler)
        R_right = _euler_xyz_to_rotmat(right_euler)
        # Ensure `g` matches the internal computation dtype (float64) to avoid
        # mixed-dtype einsum errors after projector tensors are cast to bf16/fp16.
        g = self.gripper_in_eef.to(device=device, dtype=act.dtype)
        # gripper: pos + R(euler) @ offset
        p_follow_L_grip = left_pos + torch.einsum("...ij,j->...i", R_left, g)
        p_follow_R_grip = right_pos + torch.einsum("...ij,j->...i", R_right, g)
        # EEF
        p_follow_L_eef = left_pos
        p_follow_R_eef = right_pos
        M1L, M2L = self.M1_left.to(device), self.M2_left.to(device)
        M1R, M2R = self.M1_right.to(device), self.M2_right.to(device)

        def _project_world(world_L: torch.Tensor, world_R: torch.Tensor) -> dict:
            # --- face_view (head camera, static) --------------------------------
            Kh = self.K_head.to(device)
            face_pix_L = _pinhole_project(world_L, Kh)
            face_pix_R = _pinhole_project(world_R, Kh)
            face_view = torch.cat([face_pix_L, face_pix_R], dim=-1)
            # --- left_wrist_view (camera mounted on left arm) -------------------
            preL = self.pre_transform_left.to(device)
            extL = self.extrinsics_left.to(device)
            Kl = self.K_left.to(device)
            x_l = _apply_T_batch(preL, world_L)
            x_l = torch.einsum("...ji,...j->...i", R_left, x_l - left_pos)
            x_l = _apply_T_batch(extL, x_l)
            left_wrist_pix = _pinhole_project(x_l, Kl)
            # --- right_wrist_view (camera mounted on right arm) -----------------
            preR = self.pre_transform_right.to(device)
            extR = self.extrinsics_right.to(device)
            Kr = self.K_right.to(device)
            x_r = _apply_T_batch(preR, world_R)
            x_r = torch.einsum("...ji,...j->...i", R_right, x_r - right_pos)
            x_r = _apply_T_batch(extR, x_r)
            right_wrist_pix = _pinhole_project(x_r, Kr)
            out_dtype = abs_actions.dtype
            return {
                "face_view": face_view.to(dtype=out_dtype),
                "left_wrist_view": left_wrist_pix.to(dtype=out_dtype),
                "right_wrist_view": right_wrist_pix.to(dtype=out_dtype),
            }

        world_L_grip = _apply_T_batch(M2L, _apply_T_batch(M1L, p_follow_L_grip))
        world_R_grip = _apply_T_batch(M2R, _apply_T_batch(M1R, p_follow_R_grip))
        world_L_eef = _apply_T_batch(M2L, _apply_T_batch(M1L, p_follow_L_eef))
        world_R_eef = _apply_T_batch(M2R, _apply_T_batch(M1R, p_follow_R_eef))
        if return_eef == 0:
            return _project_world(world_L_grip, world_R_grip)
        if return_eef == 1:
            return _project_world(world_L_eef, world_R_eef)
        if return_eef == 2:
            return {
                "gripper": _project_world(world_L_grip, world_R_grip),
                "eef": _project_world(world_L_eef, world_R_eef),
            }
        raise ValueError(f"Invalid return_eef={return_eef}, expected 0, 1, or 2")
