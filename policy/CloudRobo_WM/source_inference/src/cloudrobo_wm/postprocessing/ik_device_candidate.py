from __future__ import annotations
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from . import ik as frozen_ik
ARMS = frozen_ik.ARMS
GRIPPERS = frozen_ik.GRIPPERS
POLICY = dict(frozen_ik.POLICY)

class BoundedIKDevice:

    def __init__(self, fk, *, device='cuda:0', batch_backtracks=False, limits=None):
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('This candidate supports only CPU or CUDA')
        self.batch_backtracks = bool(batch_backtracks)
        self.policy = dict(POLICY, optimization_device=str(self.device))
        with torch.inference_mode(False):
            self.fk = fk.to(device=self.device, dtype=torch.float32).eval()
            for name, buffer in list(self.fk.named_buffers()):
                owner, _, leaf = name.rpartition('.')
                module = self.fk.get_submodule(owner) if owner else self.fk
                module._buffers[leaf] = buffer.detach().clone()
            bounds = frozen_ik.urdf_arm_limits(fk) if limits is None else limits
            self.lower, self.upper = tuple((torch.as_tensor(x, dtype=torch.float32, device=self.device).clone() for x in bounds))
        if self.lower.shape != (12,) or self.upper.shape != (12,) or (not (self.lower < self.upper).all()):
            raise ValueError('Invalid twelve arm joint bounds')
        self.last_accept_backtracks = None

    def __call__(self, raw_joint14, relative_eef, anchor_pose, valid=None):
        started = time.perf_counter()
        raw = np.asarray(raw_joint14)
        if raw.shape != (32, 14) or raw.dtype != np.float32 or (not np.isfinite(raw).all()):
            raise ValueError('IK input must be finite FP32 absolute [32,14] commands')
        mask = np.ones((32, 2), bool) if valid is None else np.asarray(valid, dtype=bool)
        if mask.shape != (32, 2):
            raise ValueError('Expected valid mask [32,2]')
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(self.device.type, enabled=False):
            return self._optimize(raw, relative_eef, anchor_pose, mask, started)

    @staticmethod
    def _CPU_target_input(value):
        return value.detach().cpu() if torch.is_tensor(value) else value

    def _optimize(self, raw, relative, anchor, mask, started):
        q0 = torch.from_numpy(raw.copy()).to(self.device)
        out = q0.clone()
        reasons = np.full((32, 2), 'no_accepted_step', dtype=object)
        trace = torch.full((POLICY['iterations'], 32, 2), -1, dtype=torch.int64, device=self.device)
        self.last_accept_backtracks = trace
        try:
            target_xyz, target_rot = frozen_ik.targets_from_prediction(self._CPU_target_input(relative), self._CPU_target_input(anchor))
        except ValueError as exc:
            return (raw.copy(), dict(policy=self.policy, latency_seconds=time.perf_counter() - started, fallback_reason='invalid_model_EEF_target: ' + str(exc), changed_arm_steps=0, fallback_count=64, accepted_steps=[[0, 0]] * 32, raw_retained=True))
        if target_xyz.shape != (32, 2, 3):
            raise ValueError('Target horizon differs from command horizon')
        target_xyz = target_xyz.to(self.device)
        target_rot = target_rot.to(self.device)
        finite = torch.isfinite(target_xyz).all(-1) & torch.isfinite(target_rot).all(-1).all(-1)
        lower, upper = (self.lower.reshape(2, 6), self.upper.reshape(2, 6))
        original = q0[:, ARMS].reshape(32, 2, 6)
        within = (original >= lower).all(-1) & (original <= upper).all(-1)
        eligible = torch.from_numpy(mask).to(self.device) & finite & within
        reasons[~mask] = 'masked'
        reasons[(~within).detach().cpu().numpy()] = 'raw_outside_URDF_limits'
        base = self.fk(q0).detach()
        base_error = (base[..., :3, 3] - target_xyz).square().sum(-1)
        base_angle = frozen_ik.orientation_error_degrees(base[..., :3, :3], target_rot)
        current = base_error.clone()
        accepted_count = torch.zeros((32, 2), dtype=torch.int64, device=self.device)
        lo = torch.maximum(lower[None], original - POLICY['cumulative_rad'])
        hi = torch.minimum(upper[None], original + POLICY['cumulative_rad'])
        for iteration in range(POLICY['iterations']):
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
            if self.batch_backtracks:
                out, current, accepted, first_bt = self._batch_backtrack(out, current, before, direction, pending, lo, hi, target_xyz, target_rot, base_angle)
                accepted_count += accepted.to(torch.int64)
                trace[iteration] = torch.where(accepted, first_bt, trace[iteration])
                pending &= ~accepted
            else:
                for bt in range(POLICY['backtracks']):
                    if not pending.any():
                        break
                    proposal = torch.maximum(lo, torch.minimum(hi, before + direction * 0.5 ** bt))
                    candidate = out.clone()
                    candidate[:, ARMS] = torch.where(pending[..., None], proposal, out[:, ARMS].reshape(32, 2, 6)).reshape(32, 12)
                    p = self.fk(candidate).detach()
                    err = (p[..., :3, 3] - target_xyz).square().sum(-1)
                    angle = frozen_ik.orientation_error_degrees(p[..., :3, :3], target_rot)
                    accept = self._accept(pending, err, angle, current, base_angle)
                    result = out[:, ARMS].reshape(32, 2, 6)
                    out[:, ARMS] = torch.where(accept[..., None], proposal, result).reshape(32, 12)
                    current = torch.where(accept, err, current)
                    accepted_count += accept.to(torch.int64)
                    trace[iteration] = torch.where(accept, bt, trace[iteration])
                    pending &= ~accept
            eligible &= grad_ok & ~pending
        final = self.fk(out).detach()
        final_error = (final[..., :3, 3] - target_xyz).square().sum(-1)
        final_angle = frozen_ik.orientation_error_degrees(final[..., :3, :3], target_rot)
        modified = accepted_count > 0
        safe = modified & (final_error < base_error - POLICY['minimum_position_improvement_m2']) & (final_angle <= base_angle + POLICY['orientation_slack_degrees'])
        out[:, ARMS] = torch.where(safe[..., None], out[:, ARMS].reshape(32, 2, 6), original).reshape(32, 12)
        accepted_count = torch.where(safe, accepted_count, torch.zeros_like(accepted_count))
        safe_CPU = safe.detach().cpu().numpy()
        reasons[safe_CPU] = 'accepted'
        result = out.detach().cpu().numpy().astype(raw.dtype, copy=True)
        assert np.array_equal(result[:, GRIPPERS], raw[:, GRIPPERS])
        assert (out[:, ARMS] - q0[:, ARMS]).abs().max() <= POLICY['cumulative_rad'] + 2e-07
        final = self.fk(out).detach()
        final_error = (final[..., :3, 3] - target_xyz).square().sum(-1)
        final_angle = frozen_ik.orientation_error_degrees(final[..., :3, :3], target_rot)
        return (result, dict(policy=self.policy, latency_seconds=time.perf_counter() - started, changed_arm_steps=int(safe.sum()), fallback_count=int((~safe).sum()), accepted_steps=accepted_count.detach().cpu().tolist(), fallback_reason=reasons.tolist(), raw_retained=not bool(safe.any()), before_position_error_m=torch.sqrt(base_error).detach().cpu().tolist(), after_position_error_m=torch.sqrt(final_error).detach().cpu().tolist(), before_orientation_error_degrees=base_angle.detach().cpu().tolist(), after_orientation_error_degrees=final_angle.detach().cpu().tolist(), max_joint_change_rad=float((out[:, ARMS] - q0[:, ARMS]).abs().max())))

    @staticmethod
    def _accept(pending, err, angle, current, base_angle):
        return pending & torch.isfinite(err) & torch.isfinite(angle) & (err < current - POLICY['minimum_position_improvement_m2']) & (angle <= base_angle + POLICY['orientation_slack_degrees'])

    def _batch_backtrack(self, out, current, before, direction, pending, lo, hi, target_xyz, target_rot, base_angle):
        scales = torch.tensor([0.5 ** bt for bt in range(POLICY['backtracks'])], dtype=out.dtype, device=self.device).reshape(-1, 1, 1, 1)
        proposals = torch.maximum(lo[None], torch.minimum(hi[None], before[None] + direction[None] * scales))
        candidates = out[None].expand(POLICY['backtracks'], -1, -1).clone()
        candidates[:, :, ARMS] = torch.where(pending[None, ..., None], proposals, out[:, ARMS].reshape(32, 2, 6)[None]).reshape(POLICY['backtracks'], 32, 12)
        p = self.fk(candidates).detach()
        err = (p[..., :3, 3] - target_xyz[None]).square().sum(-1)
        angle = frozen_ik.orientation_error_degrees(p[..., :3, :3], target_rot[None])
        accept = self._accept(pending[None], err, angle, current[None], base_angle[None])
        accepted = accept.any(0)
        first_bt = accept.to(torch.int64).argmax(0)
        chosen_proposal = proposals.gather(0, first_bt[None, ..., None].expand(1, 32, 2, 6)).squeeze(0)
        chosen_error = err.gather(0, first_bt[None]).squeeze(0)
        out[:, ARMS] = torch.where(accepted[..., None], chosen_proposal, out[:, ARMS].reshape(32, 2, 6)).reshape(32, 12)
        current = torch.where(accepted, chosen_error, current)
        return (out, current, accepted, first_bt)
__all__ = ['BoundedIKDevice', 'POLICY']
