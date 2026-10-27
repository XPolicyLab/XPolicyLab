"""Exact pure rotation helpers extracted from the PPU source snapshot."""
import torch
from wam.datasets.lerobot.utils.rotation import matrix_to_quaternion, quaternion_to_axis_angle

def _skew_from_vector(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)

def _rot6d_columns_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Invert the compact wrist producer's two-column rotation encoding."""
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:6]
    first = torch.nn.functional.normalize(first, dim=-1)
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)

def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    rv = rotvec.to(torch.float32)
    theta = rv.norm(dim=-1, keepdim=True)
    axis = rv / theta.clamp_min(1e-8)
    k = _skew_from_vector(axis)
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device).expand(*rv.shape[:-1], 3, 3)
    theta_m = theta.unsqueeze(-1)
    return eye + torch.sin(theta_m) * k + (1.0 - torch.cos(theta_m)) * (k @ k)

def _matrix_to_rotvec(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle without collapsing pi turns.

    The trace/vee formula is singular at 180 degrees: the skew part is zero,
    so exact pi rotations were previously projected to a zero rotation vector.
    The quaternion branch implementation is well-conditioned both near zero
    and near pi and uses the same WXYZ convention as the rest of this loader.
    """
    r = rotation.to(torch.float32)
    return quaternion_to_axis_angle(matrix_to_quaternion(r))
