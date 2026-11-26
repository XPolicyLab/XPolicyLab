"""Optional action-only gripper tightening for RoboDojo policy evaluation."""
from __future__ import annotations

import numpy as np


class GripperCloseHold:
    """Track raw closing/opening trends independently per environment and hand.

    Inputs are decoded RoboDojo [T, 16] commands, with 0=closed and 1=open.
    Use the full predicted non-anchor horizon to anticipate closing. Future
    opening never releases the flag early: only selected executable commands
    update persistent state. Never feed corrected commands back into trend
    detection or replace observed proprioception with them.
    """

    def __init__(self, offset: float = 0.0, motion_tolerance: float = 0.01):
        self.offset = float(offset)
        self.motion_tolerance = float(motion_tolerance)
        if not np.isfinite(self.offset) or not 0 <= self.offset <= 1:
            raise ValueError("gripper_close_offset must be finite and in [0, 1]")
        if not np.isfinite(self.motion_tolerance) or not 0 <= self.motion_tolerance < 1:
            raise ValueError("gripper_motion_tolerance must be finite and in [0, 1)")
        self._states: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self._states.clear()

    def apply(self, chunk: np.ndarray, initial_grippers: np.ndarray, env_idx: int,
              *, executable_indices=None) -> np.ndarray:
        if self.offset == 0:
            return chunk  # Preserve the baseline path exactly, including out-of-range values.
        source = np.asarray(chunk)
        if source.ndim != 2 or source.shape[1] != 16:
            raise ValueError(f"Expected decoded [T, 16] action chunk, got {source.shape}")
        if not np.issubdtype(source.dtype, np.floating):
            raise ValueError("Decoded action chunk must have floating-point dtype")
        initial = np.asarray(initial_grippers, dtype=np.float64)
        if initial.shape != (2,) or not np.isfinite(initial).all():
            raise ValueError("initial_grippers must contain two finite observed apertures")
        if not np.isfinite(source[:, [7, 15]]).all():
            raise ValueError("Predicted gripper commands must be finite")
        if not len(source):
            return chunk
        if executable_indices is None:
            executable_indices = np.arange(len(source))
        indices = np.asarray(executable_indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(source))
                or np.any(np.diff(indices) <= 0)):
            raise ValueError("executable_indices must be strictly increasing valid integer indices")
        if not len(indices):
            return chunk
        env_idx = int(env_idx)
        previous = self._states.get(env_idx)
        if previous is None:
            reference = np.clip(initial, 0.0, 1.0)
            holding = np.zeros(2, dtype=bool)
        else:
            reference, holding = (a.copy() for a in previous)
        # Full-horizon lookahead can set a flag, but cannot clear one. Scan
        # relative to the raw upper envelope so slow closing accumulates.
        closing_reference = reference.copy()
        anticipated_closing = np.zeros(2, dtype=bool)
        for raw in np.clip(source[:, [7, 15]], 0.0, 1.0).astype(np.float64):
            anticipated_closing |= raw < closing_reference - self.motion_tolerance
            closing_reference = np.maximum(closing_reference, raw)
        holding |= anticipated_closing
        result = source.copy()
        for index in indices:
            row = result[index].copy()
            raw = np.clip(row[[7, 15]].astype(np.float64), 0.0, 1.0)
            movement = raw - reference
            closing = movement < -self.motion_tolerance
            opening = movement > self.motion_tolerance
            holding[closing] = True
            holding[opening] = False
            reference[closing | opening] = raw[closing | opening]
            # Follow extrema so sub-tolerance movements can accumulate and
            # opening is measured from the raw closure minimum, not its bias.
            reference = np.where(holding, np.minimum(reference, raw),
                                 np.maximum(reference, raw))
            values = row[[7, 15]].copy()
            values[holding] = np.clip(raw[holding] - self.offset, 0.0, 1.0)
            row[[7, 15]] = values
            result[index] = row
        self._states[env_idx] = (reference, holding)
        return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offset', type=float, default=0.0)
    parser.add_argument('--motion-tolerance', type=float, default=0.01)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    control = GripperCloseHold(args.offset, args.motion_tolerance)
    config = dict(enabled=bool(control.offset), gripper_close_offset=control.offset,
                  gripper_motion_tolerance=control.motion_tolerance,
                  rule='full_nonanchor_horizon_preclose_executable_open_release_v1',
                  space='decoded_0_1_aperture', state_unchanged=True)
    payload = json.dumps(config, indent=2) + '\n'
    if args.output:
        args.output.write_text(payload)
    print(payload, end='')
