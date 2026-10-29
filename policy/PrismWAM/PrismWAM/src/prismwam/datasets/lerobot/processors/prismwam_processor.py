from typing import Any, Dict, List, Optional

import torch

from ..utils.normalizer import LinearNormalizer


class PrismWAMProcessor:
    """LIBERO action/state normalization: per-dimension min/max to [-1, 1]."""

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_output_dim: int,
        proprio_output_dim: int,
        action_state_merger,
        delta_action_dim_mask: Optional[Dict[str, List[bool]]] = None,
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim
        self._is_train = None
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)
        self._normalizer = None
        if delta_action_dim_mask is None:
            self.delta_action_dim_mask = None
        else:
            action_meta = self.shape_meta["action"]
            expected_keys = [m["key"] for m in action_meta]
            provided_keys = list(delta_action_dim_mask.keys())
            if set(provided_keys) != set(expected_keys):
                raise ValueError(
                    f"`delta_action_dim_mask` keys mismatch. Expected {expected_keys}, got {provided_keys}."
                )

            self.delta_action_dim_mask = {}
            for meta in action_meta:
                key = meta["key"]
                expected_dim = meta["shape"]
                mask = delta_action_dim_mask[key]
                if len(mask) != expected_dim:
                    raise ValueError(
                        f"`delta_action_dim_mask[{key}]` length must be {expected_dim}, got {len(mask)}."
                    )
                self.delta_action_dim_mask[key] = torch.as_tensor(
                    mask, dtype=torch.bool
                )

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError(
                "is_train has not been set. Please call train() and eval() first."
            )
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError(
                "normalizer has not been set. Please call set_normalizer_from_stats() first."
            )
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any]):
        self._normalizer = LinearNormalizer(
            shape_meta=self.shape_meta,
            stats=dataset_stats,
        )

    def action_state_transform(self, batch):
        """Check that action and state match the configured shapes."""
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta["raw_shape"] == meta_shape, (
                    f"Action key {k} shape {actual_shape} mismatch with meta raw shape {meta['raw_shape']} / shape {meta_shape}."
                )

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta["raw_shape"] == meta_shape, (
                f"State key {k} shape {actual_shape} mismatch with meta raw shape {meta['raw_shape']} / shape {meta_shape}."
            )

        return batch
