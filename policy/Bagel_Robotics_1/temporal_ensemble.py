"""Blend aligned absolute EE16 predictions, keeping the new chunk dominant."""
import numpy as np


class TemporalEnsemble:
    def __init__(self, horizon=32, replan_steps=16, new_weight=0.9, old_weight=0.1):
        if (horizon, replan_steps) != (32, 16):
            raise ValueError('This two-chunk ensemble requires horizon 32 and replan 16')
        if (new_weight, old_weight) != (0.9, 0.1):
            raise ValueError('This evaluation requires new=0.9, old=0.1')
        self.new_weight, self.old_weight = new_weight, old_weight
        self.previous = None
        self.step = 0
        self.last_info = None

    def apply(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape != (32, 16) or not np.isfinite(chunk).all():
            raise ValueError('Expected a finite [32,16] action prediction')
        new = chunk[:16]
        old = None if self.previous is None else self.previous[16:32]
        if old is None:
            output = new.copy()
        else:
            output = self.new_weight * new + self.old_weight * old
            for sl in (slice(3, 7), slice(11, 15)):
                qnew, qold = new[:, sl].copy(), old[:, sl].copy()
                nnew = np.linalg.norm(qnew, axis=-1, keepdims=True)
                nold = np.linalg.norm(qold, axis=-1, keepdims=True)
                if np.any(nnew < 1e-8) or np.any(nold < 1e-8):
                    raise ValueError('Cannot ensemble a zero-norm quaternion')
                qnew /= nnew
                qold /= nold
                qold *= np.where(np.sum(qnew * qold, axis=-1, keepdims=True) < 0, -1, 1)
                blended = self.new_weight * qnew + self.old_weight * qold
                output[:, sl] = blended / np.linalg.norm(blended, axis=-1, keepdims=True)
        self.last_info = dict(start_step=self.step, horizon=32, executed=16,
                              new_weight=self.new_weight, old_weight=self.old_weight,
                              overlap=old is not None, new_chunk=chunk.copy(),
                              old_aligned=None if old is None else old.copy(),
                              output=output.copy())
        # Keep raw predictions: blending must not recursively change the old target.
        self.previous = chunk.copy()
        self.step += 16
        return output
