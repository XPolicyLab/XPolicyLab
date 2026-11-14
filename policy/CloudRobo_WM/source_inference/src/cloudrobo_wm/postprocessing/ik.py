from __future__ import annotations
import math
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from .robotwin_fk import axis_angle_to_matrix_torch
ARMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPERS = (6, 13)
POLICY = dict(iterations=10, backtracks=8, step_rad=0.02, cumulative_rad=0.05, orientation_slack_degrees=1.0, minimum_position_improvement_m2=1e-12, target_source='same_prediction_relative_EEF_and_fixed_observed_anchor', optimization_dtype='float32', optimization_device='cpu', smoothing=False)

def quaternion_matrix(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1).reshape(q.shape[:-1] + (3, 3))

def targets_from_prediction(relative_eef, anchor_pose):
    relative = torch.as_tensor(relative_eef, dtype=torch.float32).clone()
    anchor = torch.as_tensor(anchor_pose, dtype=torch.float32).clone()
    if relative.ndim != 3 or relative.shape[1:] != (2, 6) or anchor.shape != (2, 7):
        raise ValueError('Expected relative EEF [T,2,6] and fixed anchor [2,7]')
    if not torch.isfinite(relative).all() or not torch.isfinite(anchor).all():
        raise ValueError('Nonfinite EEF prediction/anchor')
    if (torch.linalg.vector_norm(anchor[:, 3:], dim=-1) < 1e-06).any():
        raise ValueError('Zero anchor quaternion')
    xyz = relative[:, :, :3] + anchor[None, :, :3]
    rot = quaternion_matrix(anchor[:, 3:])[None] @ axis_angle_to_matrix_torch(relative[:, :, 3:])
    return (xyz, rot)

def urdf_arm_limits(fk):
    records = {j.attrib['name']: j for j in ET.parse(fk.urdf_path).getroot().findall('joint')}
    lower, upper = ([], [])
    for names in fk.arm_joint_names:
        for name in names:
            joint = records[name]
            if joint.attrib['type'] == 'continuous':
                lower.append(-float('inf'))
                upper.append(float('inf'))
                continue
            limit = joint.find('limit')
            if limit is None or 'lower' not in limit.attrib or 'upper' not in limit.attrib:
                raise ValueError('Missing finite URDF limits for ' + name)
            lo, hi = (float(limit.attrib['lower']), float(limit.attrib['upper']))
            if not math.isfinite(lo + hi) or lo >= hi:
                raise ValueError('Invalid URDF limits ' + name)
            lower.append(lo)
            upper.append(hi)
    return (torch.tensor(lower, dtype=torch.float32), torch.tensor(upper, dtype=torch.float32))

def orientation_error_degrees(rotation, target):
    relative = rotation.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))

class BoundedIK:

    def __init__(self, fk, *, limits=None):
        with torch.inference_mode(False):
            self.fk = fk.to(device='cpu', dtype=torch.float32).eval()
            for name, buffer in list(self.fk.named_buffers()):
                owner, _, leaf = name.rpartition('.')
                module = self.fk.get_submodule(owner) if owner else self.fk
                module._buffers[leaf] = buffer.detach().clone()
            self.lower, self.upper = urdf_arm_limits(fk) if limits is None else tuple((torch.as_tensor(x, dtype=torch.float32).clone() for x in limits))
        if self.lower.shape != (12,) or self.upper.shape != (12,) or (not (self.lower < self.upper).all()):
            raise ValueError('Invalid twelve arm joint bounds')

    def __call__(self, raw_joint14, relative_eef, anchor_pose, valid=None):
        started = time.perf_counter()
        raw = np.asarray(raw_joint14)
        if raw.shape != (32, 14) or raw.dtype != np.float32 or (not np.isfinite(raw).all()):
            raise ValueError('IK input must be finite FP32 absolute [32,14] commands')
        mask = np.ones((32, 2), bool) if valid is None else np.asarray(valid, dtype=bool)
        if mask.shape != (32, 2):
            raise ValueError('Expected valid mask [32,2]')
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast('cpu', enabled=False):
            return self._optimize(raw, relative_eef, anchor_pose, mask, started)

    def _optimize(self, raw, relative, anchor, mask, started):
        q0 = torch.from_numpy(raw.copy())
        out = q0.clone()
        reasons = np.full((32, 2), 'no_accepted_step', dtype=object)
        try:
            target_xyz, target_rot = targets_from_prediction(relative, anchor)
        except ValueError as exc:
            return (raw.copy(), dict(policy=POLICY, latency_seconds=time.perf_counter() - started, fallback_reason='invalid_model_EEF_target: ' + str(exc), changed_arm_steps=0, fallback_count=64, accepted_steps=[[0, 0]] * 32, raw_retained=True))
        if target_xyz.shape != (32, 2, 3):
            raise ValueError('Target horizon differs from command horizon')
        finite = torch.isfinite(target_xyz).all(-1) & torch.isfinite(target_rot).all(-1).all(-1)
        lower = self.lower.reshape(2, 6)
        upper = self.upper.reshape(2, 6)
        original = q0[:, ARMS].reshape(32, 2, 6)
        within = (original >= lower).all(-1) & (original <= upper).all(-1)
        eligible = torch.from_numpy(mask) & finite & within
        reasons[~mask] = 'masked'
        reasons[(~within).numpy()] = 'raw_outside_URDF_limits'
        base = self.fk(q0).detach()
        base_error = (base[..., :3, 3] - target_xyz).square().sum(-1)
        base_angle = orientation_error_degrees(base[..., :3, :3], target_rot)
        current = base_error.clone()
        accepted_count = torch.zeros((32, 2), dtype=torch.int64)
        lo = torch.maximum(lower[None], original - POLICY['cumulative_rad'])
        hi = torch.minimum(upper[None], original + POLICY['cumulative_rad'])
        for _ in range(POLICY['iterations']):
            if not eligible.any():
                break
            q = out.detach().clone().requires_grad_(True)
            pose = self.fk(q)
            errors = (pose[..., :3, 3] - target_xyz).square().sum(-1)
            gradient, = torch.autograd.grad((errors * eligible).sum(), q)
            grad = gradient[:, ARMS].reshape(32, 2, 6)
            grad_ok = torch.isfinite(grad).all(-1) & (grad.abs().amax(-1) > 1e-12)
            direction = -grad / grad.abs().amax(-1, keepdim=True).clamp_min(1e-12) * POLICY['step_rad']
            before = out[:, ARMS].reshape(32, 2, 6).clone()
            pending = eligible & grad_ok
            for bt in range(POLICY['backtracks']):
                if not pending.any():
                    break
                proposal = torch.maximum(lo, torch.minimum(hi, before + direction * 0.5 ** bt))
                candidate = out.clone()
                candidate[:, ARMS] = torch.where(pending[..., None], proposal, out[:, ARMS].reshape(32, 2, 6)).reshape(32, 12)
                p = self.fk(candidate).detach()
                err = (p[..., :3, 3] - target_xyz).square().sum(-1)
                angle = orientation_error_degrees(p[..., :3, :3], target_rot)
                accept = pending & torch.isfinite(err) & torch.isfinite(angle) & (err < current - POLICY['minimum_position_improvement_m2']) & (angle <= base_angle + POLICY['orientation_slack_degrees'])
                result = out[:, ARMS].reshape(32, 2, 6)
                out[:, ARMS] = torch.where(accept[..., None], proposal, result).reshape(32, 12)
                current = torch.where(accept, err, current)
                accepted_count += accept.to(torch.int64)
                pending &= ~accept
            eligible &= grad_ok & ~pending
        final = self.fk(out).detach()
        final_error = (final[..., :3, 3] - target_xyz).square().sum(-1)
        final_angle = orientation_error_degrees(final[..., :3, :3], target_rot)
        modified = accepted_count > 0
        safe = modified & (final_error < base_error - POLICY['minimum_position_improvement_m2']) & (final_angle <= base_angle + POLICY['orientation_slack_degrees'])
        out[:, ARMS] = torch.where(safe[..., None], out[:, ARMS].reshape(32, 2, 6), original).reshape(32, 12)
        accepted_count = torch.where(safe, accepted_count, torch.zeros_like(accepted_count))
        reasons[safe.numpy()] = 'accepted'
        result = out.detach().numpy().astype(raw.dtype, copy=True)
        assert np.array_equal(result[:, GRIPPERS], raw[:, GRIPPERS])
        assert (out[:, ARMS] - q0[:, ARMS]).abs().max() <= POLICY['cumulative_rad'] + 2e-07
        final = self.fk(out).detach()
        final_error = (final[..., :3, 3] - target_xyz).square().sum(-1)
        final_angle = orientation_error_degrees(final[..., :3, :3], target_rot)
        return (result, dict(policy=POLICY, latency_seconds=time.perf_counter() - started, changed_arm_steps=int(safe.sum()), fallback_count=int((~safe).sum()), accepted_steps=accepted_count.tolist(), fallback_reason=reasons.tolist(), raw_retained=not bool(safe.any()), before_position_error_m=torch.sqrt(base_error).tolist(), after_position_error_m=torch.sqrt(final_error).tolist(), before_orientation_error_degrees=base_angle.tolist(), after_orientation_error_degrees=final_angle.tolist(), max_joint_change_rad=float((out[:, ARMS] - q0[:, ARMS]).abs().max())))

def apply_policy_prediction(policy, raw_joint14):
    prediction = policy.last_prediction
    if not isinstance(prediction, dict) or prediction.get('eef_frame') != 'raw_fl_link6_fr_link6_world':
        raise ValueError('Expected the original policy decoded EEF frame')
    if not np.array_equal(np.asarray(prediction['joint14'], np.float32), raw_joint14):
        raise ValueError('Joint commands and EEF target must belong to the same prediction')
    if not hasattr(policy, '_bounded_ik'):
        original = policy.normalizer._load_fk()
        from .robotwin_fk import RobotWinUrdfFK
        fk = RobotWinUrdfFK(original.urdf_path, original.embodiment_config_path, expected_urdf_sha256=original.urdf_sha256)
        policy._bounded_ik = BoundedIK(fk)
    return policy._bounded_ik(raw_joint14, prediction['eef_relative'], prediction['anchor_pose'])
