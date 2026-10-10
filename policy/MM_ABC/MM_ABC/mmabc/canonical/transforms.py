"""Map absolute state/action vectors to masked model targets and back."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmabc.canonical.layout import CanonicalLayout
from mmabc.canonical import rotation as rot


@dataclass(frozen=True)
class EmbodimentSpec:
    """Per-embodiment facts needed to build targets, from embodiment.json."""

    embodiment_tag: str
    fps: float
    state_valid: np.ndarray  # (80,) bool
    action_valid: np.ndarray  # (80,) bool

    @classmethod
    def from_meta(cls, embodiment_json: dict, fps: float | None = None) -> EmbodimentSpec:
        canon = embodiment_json["canonical"]
        total = int(canon["total_dim"])
        # State may have its own width when action and proprio are decoupled;
        # defaults to the action width for legacy shared-layout embodiments.
        state_total = int(canon.get("state_total_dim", total))
        state = np.zeros(state_total, dtype=bool)
        action = np.zeros(total, dtype=bool)
        if "state_valid" in canon:
            state[:] = np.asarray(canon["state_valid"], dtype=bool)
        else:
            state[np.asarray(canon["state_valid_dims"], dtype=int)] = True
        if "action_valid" in canon:
            action[:] = np.asarray(canon["action_valid"], dtype=bool)
        else:
            action[np.asarray(canon["action_valid_dims"], dtype=int)] = True
        return cls(
            embodiment_tag=str(canon.get("embodiment_tag", embodiment_json.get("embodiment_tag", "?"))),
            fps=float(fps if fps is not None else embodiment_json.get("record_frequency", 30.0)),
            state_valid=state,
            action_valid=action,
        )


class TargetBuilder:
    """Builds model-space regression targets and masks for one embodiment.

    The static part of the mask (which depends only on the embodiment, the
    control convention and the layout) is precomputed once per embodiment, so
    the per-sample path is arithmetic on already-known index arrays.
    """

    def __init__(
        self,
        layout: CanonicalLayout,
        spec: EmbodimentSpec,
        *,
        reference_frame: str = "base",
    ) -> None:
        if reference_frame not in layout.reference_frames:
            raise ValueError(f"unknown reference frame {reference_frame!r}")
        self.layout = layout
        self.spec = spec
        self.reference_frame = reference_frame
        self._static_masks = {
            name: self._build_static_mask(name) for name in layout.action_types
        }
        self.available_action_types = layout.available_action_types(spec.action_valid)
        if not self.available_action_types:
            raise ValueError(
                f"embodiment {spec.embodiment_tag} has no supervisable action convention"
            )

    def _build_static_mask(self, action_type: str) -> np.ndarray:
        layout = self.layout
        mask = np.zeros(layout.total_dim, dtype=bool)
        keep = layout.action_type_mask(action_type)

        for seg in layout.segments:
            if seg.kind == "reserved":
                continue
            if not keep[seg.start]:
                continue
            a_ok = self.spec.action_valid[seg.slice]
            s_ok = self.spec.state_valid[seg.slice]

            if seg.kind == "eef_rot":
                # A rotation increment needs the whole 6D frame at both ends.
                if a_ok.all() and s_ok.all():
                    mask[seg.target_slice] = True
            elif seg.kind == "eef_pos":
                if a_ok.all() and s_ok.all():
                    mask[seg.slice] = True
            elif seg.kind == "base_pose":
                if a_ok.all() and s_ok.all():
                    mask[seg.slice] = True
            elif seg.is_delta:
                # Elementwise increment: per-dim intersection is enough.
                mask[seg.slice] = a_ok & s_ok
            else:
                mask[seg.slice] = a_ok
        return mask & self.layout.target_support()

    def static_mask(self, action_type: str) -> np.ndarray:
        return self._static_masks[action_type]

    def aux_is_active(self, action_type: str) -> bool:
        aux = self.layout.head("aux")
        return bool(self._static_masks[action_type][aux.slice].any())

    def build(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        *,
        action_type: str,
        valid_steps: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert one sample to model space."""
        layout = self.layout
        T = actions.shape[0]
        state = np.asarray(state, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)

        target = np.zeros((T, layout.total_dim), dtype=np.float64)
        keep = layout.action_type_mask(action_type)

        for seg in layout.segments:
            if seg.kind == "reserved" or not keep[seg.start]:
                continue
            sl = seg.slice

            if seg.kind == "eef_rot":
                r_t = rot.rot6d_to_matrix(state[sl])
                r_k = rot.rot6d_to_matrix(actions[:, sl])
                rel = rot.relative_rotation(np.broadcast_to(r_t, r_k.shape), r_k)
                target[:, seg.target_slice] = rot.matrix_to_axis_angle(rel)

            elif seg.kind == "eef_pos":
                delta = actions[:, sl] - state[sl]
                if self.reference_frame == "eef_local":
                    # Express the translation in the anchor end-effector frame.
                    rot_seg = self._paired_rotation_segment(seg)
                    if rot_seg is not None:
                        r_t = rot.rot6d_to_matrix(state[rot_seg.slice])
                        delta = delta @ r_t  # inv(R) @ d == d @ R for row vectors
                target[:, sl] = delta

            elif seg.kind == "base_pose":
                # Planar pose expressed in the anchor heading frame.
                dxy = actions[:, sl][:, :2] - state[sl][:2]
                yaw_t = state[sl][2]
                c, s = np.cos(yaw_t), np.sin(yaw_t)
                target[:, seg.start + 0] = c * dxy[:, 0] + s * dxy[:, 1]
                target[:, seg.start + 1] = -s * dxy[:, 0] + c * dxy[:, 1]
                target[:, seg.start + 2] = rot.wrap_angle(actions[:, seg.start + 2] - yaw_t)

            elif seg.is_delta:
                target[:, sl] = actions[:, sl] - state[sl]

            else:
                target[:, sl] = actions[:, sl]

        mask = np.broadcast_to(self._static_masks[action_type], (T, layout.total_dim)).copy()
        if valid_steps is not None and valid_steps < T:
            # Once the episode ends there is no causal continuation, so drop
            # this step and everything after it.
            mask[max(valid_steps, 0) :, :] = False

        target = np.where(mask, target, 0.0)
        return target.astype(np.float32), mask

    def _paired_rotation_segment(self, pos_seg):
        """The rotation segment belonging to the same arm as a position segment."""
        prefix = pos_seg.name.rsplit("_eef_position", 1)[0]
        try:
            return self.layout.segment(f"{prefix}_eef_rotation")
        except KeyError:
            return None

    def integrate(
        self,
        state: np.ndarray,
        target: np.ndarray,
        *,
        action_type: str,
    ) -> np.ndarray:
        """Model space -> absolute storage space. Inverse of :meth:`build`.

        Used at deployment to turn predicted increments back into commands.
        Dims outside the supervised mask are filled with the anchor state so
        the caller can slice out whatever the robot actually consumes.
        """
        layout = self.layout
        T = target.shape[0]
        state = np.asarray(state, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if state.shape[-1] == layout.total_dim:
            # Shared state/action space: unsupervised dims fall back to the
            # anchor state (e.g. the far half of a 6D rotation).
            out = np.broadcast_to(state, (T, layout.total_dim)).copy()
        else:
            # Decoupled state/action: the proprio anchor is in
            # a different space, and every action segment is absolute/pass-through
            # (overwritten below), so start from zeros rather than the anchor.
            out = np.zeros((T, layout.total_dim), dtype=np.float64)
        keep = layout.action_type_mask(action_type)

        for seg in layout.segments:
            if seg.kind == "reserved" or not keep[seg.start]:
                continue
            sl = seg.slice

            if seg.kind == "eef_rot":
                r_t = rot.rot6d_to_matrix(state[sl])
                rel = rot.axis_angle_to_matrix(target[:, seg.target_slice])
                out[:, sl] = rot.matrix_to_rot6d(np.einsum("ij,tjk->tik", r_t, rel))

            elif seg.kind == "eef_pos":
                delta = target[:, sl]
                if self.reference_frame == "eef_local":
                    rot_seg = self._paired_rotation_segment(seg)
                    if rot_seg is not None:
                        r_t = rot.rot6d_to_matrix(state[rot_seg.slice])
                        delta = delta @ r_t.T
                out[:, sl] = state[sl] + delta

            elif seg.kind == "base_pose":
                yaw_t = state[sl][2]
                c, s = np.cos(yaw_t), np.sin(yaw_t)
                dx, dy = target[:, seg.start + 0], target[:, seg.start + 1]
                out[:, seg.start + 0] = state[seg.start + 0] + c * dx - s * dy
                out[:, seg.start + 1] = state[seg.start + 1] + s * dx + c * dy
                out[:, seg.start + 2] = rot.wrap_angle(yaw_t + target[:, seg.start + 2])

            elif seg.is_delta:
                out[:, sl] = state[sl] + target[:, sl]

            else:
                out[:, sl] = target[:, sl]

        return out.astype(np.float32)
