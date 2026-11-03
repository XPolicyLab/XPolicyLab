"""YAM real-world 32D relative-action transform.

Mirrors yam_openpi's `action_rep="rel"` recipe (RelativeEEFActions + DeltaActions
with mask `make_bool_mask(-20, 12)`) as a single FlexPi action_state_transform.

Port reference:
  yam_openpi/src/openpi/training/config.py:411-418  (LeRobotYAMDataConfig.create
                                                     for action_rep == "rel")
  yam_openpi/src/openpi/policies/yam_policy.py:336-381 (RelativeEEFActions)
  yam_openpi/src/openpi/transforms.py:228-248          (DeltaActions)

32D layout (matches yam_eef.STATE_LAYOUT, identical to AgiBot 32D):
  [0:3]    L_pos                      - relative (SE(3) body-frame)
  [3:9]    L_rot6d  (row convention)  - relative (SE(3) body-frame)
  [9:12]   R_pos                      - relative
  [12:18]  R_rot6d                    - relative
  [18:20]  L_grip, R_grip             - **absolute, pass-through**
  [20:32]  L_joint(6), R_joint(6)     - delta (scalar subtraction)

`state_dim=34` variant (mixed yam/mm2 + AgiBot runs, see mixed_embodiment_align.py):
  identical except the joint block is [20:34] — L_joint(7), R_joint(7).

Anchor (`anchor` ctor arg):
  "first" (default, matches yam_openpi): base = state[..., 0, :] — the FIRST
       obs frame in the window. For a sample starting at time t, the dataloader
       emits proprio = state[t..t+T_obs-1] (because past_obs_size=0 in
       BaseLerobotDataset), so state[..., 0, :] = state[t] = "current obs at
       sample start". Action targets state[t+1..t+T_act], so each action step
       is encoded relative to the same anchor (= "current"). UMI / diffusion-
       policy convention; what yam_openpi's RelativeEEFActions does.
  "last" (legacy FlexPi behavior pre-2026-05-09): base = state[..., T_obs-1, :]
       — the LAST obs frame in the window, which lands at the SAME timestep as
       action[T_act-1] when T_obs == T_act + 1. action[T_act-1] then encodes to
       (≈)0 and the rest of the chunk encodes to "where to back-step from the
       end". Kept for backwards compatibility with anything that was trained
       under the pre-2026-05-09 default; not recommended for new runs.

Stats-calc path (T_obs=1):
  Both anchor settings produce the same result: base = state[..., 0, :] (the
  only frame). No special-casing needed.

Why rot6d on relative (and not delta-per-step) is safe here:
  Same base for the entire action chunk (UMI / diffusion-policy convention).
  Chunk-end (t = T_act - 1) at 30 Hz / horizon 32 ≈ 1s lookahead — relative
  rotations have meaningful magnitude (std O(0.1+)), avoiding the small-angle
  SNR collapse that broke the LIBERO 95975/97125 rot6d-on-per-step-delta
  experiments. (Those ran through the rel-6D / absolute-lookahead transforms,
  removed once no config selected them; `git log -- '*absolute_pose_lookahead*'`
  recovers them.)

Why the asymmetric joint treatment (delta) vs EEF (SE(3) relative):
  Joints are scalars in a Euclidean space — additive delta is the natural
  relative repr. EEFs live on SE(3); SE(3) relative is the natural one
  there. yam_openpi made this exact choice; we replicate.
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import torch

from ..utils.yam_eef import STATE_DIM, STATE_LAYOUT, mat_to_pose9d, pose9d_to_mat


# Contiguous EEF blocks per arm: [pos(3); rot6d(6)] = 9 dims.
_LEFT_EEF_SLICE = slice(STATE_LAYOUT["left_pos"].start, STATE_LAYOUT["left_rot6d"].stop)
_RIGHT_EEF_SLICE = slice(STATE_LAYOUT["right_pos"].start, STATE_LAYOUT["right_rot6d"].stop)
# Contiguous joint block (both arms): [20:32].
_JOINT_SLICE = slice(STATE_LAYOUT["left_joint"].start, STATE_LAYOUT["right_joint"].stop)


_AnchorArg = Literal["first", "last"]


class Yam32DRelativeAction:
    """YAM 32D action: EEF SE(3) body-frame relative + joint scalar delta + gripper absolute.

    See module docstring for the anchor convention. Default `anchor="first"` is
    UMI / yam_openpi convention (anchor = state at sample-start time t).

    Forward (training, anchor="first"):
        base = state[..., 0, :]                              # first obs frame = state[t]
        For each k in [0, T_act):
            T_baseL = pose9d_to_mat(base[0:9])
            T_baseR = pose9d_to_mat(base[9:18])
            action[k, 0:9]   = mat_to_pose9d(inv(T_baseL) @ pose9d_to_mat(action[k, 0:9]))
            action[k, 9:18]  = mat_to_pose9d(inv(T_baseR) @ pose9d_to_mat(action[k, 9:18]))
            action[k, 20:32] -= base[20:32]
        # gripper [18:20] untouched
        # state untouched (still 32D absolute throughout)

    Backward (eval inverse): same base; multiply (vs invert) for the SE(3) blocks
    and add (vs subtract) for the joint block.
    """

    def __init__(self, anchor: _AnchorArg = "first", state_dim: int = STATE_DIM) -> None:
        if anchor not in ("first", "last"):
            raise ValueError(f"anchor must be 'first' or 'last', got {anchor!r}")
        # Only the joint block widens at 34D; EEF [0:18] and grippers [18:20] match.
        if state_dim not in (STATE_DIM, 34):
            raise ValueError(f"state_dim must be {STATE_DIM} or 34, got {state_dim}")
        self.anchor: _AnchorArg = anchor
        self.state_dim = int(state_dim)
        self.joint_slice = slice(_JOINT_SLICE.start, self.state_dim)

    def set_shape_meta(self, shape_meta) -> None:
        pass

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._apply(batch, backward=False)

    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._apply(batch, backward=True)

    def _apply(self, batch, *, backward: bool):
        if "state" not in batch:
            raise ValueError("Yam32DRelativeAction requires 'state' in batch (used as relative-base anchor)")
        state = batch["state"]["default"]  # [..., T_obs, state_dim]
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"state last dim must be {self.state_dim}, got {tuple(state.shape)}")
        T_obs = state.shape[-2]
        # Anchor selection. "first" = state at sample-start time (UMI / yam_openpi
        # convention; what the user-facing 32D-rel YAM config uses). "last" =
        # legacy FlexPi behavior (state at end-of-window). T_obs=1 collapses to
        # the same frame either way, so the stats-calc path is unaffected.
        anchor_idx = 0 if self.anchor == "first" else T_obs - 1
        base = state[..., anchor_idx, :]  # [..., 32]

        if "action" in batch:
            action = batch["action"]["default"]  # [..., T_act, state_dim]
            if action.shape[-1] != self.state_dim:
                raise ValueError(f"action last dim must be {self.state_dim}, got {tuple(action.shape)}")
            batch["action"]["default"] = self._encode(
                action, base, backward=backward, joint_slice=self.joint_slice)
        # state is intentionally NOT modified — proprio stays absolute end-to-end.
        return batch

    @staticmethod
    def _encode(action: torch.Tensor, base: torch.Tensor, *, backward: bool,
                joint_slice: slice = _JOINT_SLICE) -> torch.Tensor:
        """action: torch [..., T_act, D]; base: torch [..., D] → torch [..., T_act, D].

        Numpy round-trip (float64 internal) — keeps math byte-equivalent to yam_eef.py
        and yam_openpi's RelativeEEFActions / DeltaActions. Runs CPU-side in the dataloader
        worker, where the round-trip cost is negligible.
        """
        device, dtype = action.device, action.dtype
        a64 = action.detach().cpu().numpy().astype(np.float64, copy=True)  # [..., T_act, 32]
        b64 = base.detach().cpu().numpy().astype(np.float64, copy=False)    # [..., 32]

        # SE(3) body-frame relative on the two EEF blocks.
        for sl in (_LEFT_EEF_SLICE, _RIGHT_EEF_SLICE):
            base_T = pose9d_to_mat(b64[..., sl])                    # [..., 4, 4]
            act_T  = pose9d_to_mat(a64[..., sl])                    # [..., T_act, 4, 4]
            base_T_b = base_T[..., None, :, :]                      # [..., 1, 4, 4]  (broadcast on T_act)
            if not backward:
                converted_T = np.linalg.inv(base_T_b) @ act_T       # T_rel = inv(T_base) @ T_act
            else:
                converted_T = base_T_b @ act_T                      # T_abs = T_base @ T_rel
            a64[..., sl] = mat_to_pose9d(converted_T)

        # Joint scalar delta on the joint block ([20:32] at 32D, [20:34] at 34D);
        # gripper [18:20] untouched.
        base_joint = b64[..., joint_slice][..., None, :]            # [..., 1, n_joint]  (broadcast on T_act)
        if not backward:
            a64[..., joint_slice] = a64[..., joint_slice] - base_joint
        else:
            a64[..., joint_slice] = a64[..., joint_slice] + base_joint

        return torch.from_numpy(a64).to(device=device, dtype=dtype)
