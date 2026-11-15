"""Online RGB preparation for modelzoo's frozen DA3 backbone."""

from collections.abc import Mapping

import numpy as np
import torch
from modelzoo.models.vision_encoders import da3

from soma.data import conditions
from soma.inference import config as inference_config


class DA3ConditionEncoder:
    """Encode the latest frames from configured physical cameras."""

    def __init__(
        self,
        config: inference_config.GeometryConfig,
        *,
        feature_dim: int,
        device: str | torch.device,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.layers = tuple(sorted(set(config.layer_map.values())))

        self.encoder = da3.FrozenDA3Backbone(
            config.config_path,
            config.checkpoint_path,
            self.layers,
        )
        if self.encoder.feature_dim != feature_dim:
            raise ValueError(
                "DA3 checkpoint width differs from "
                "action condition feature_dim"
            )
        self.encoder.to(device=self.device, dtype=torch.bfloat16)

    @torch.no_grad()
    def encode(
        self,
        views: Mapping[str, object],
    ) -> dict[int, torch.Tensor]:
        """Return per-layer geometry tokens for one online observation."""
        missing = sorted(
            {
                view.source
                for view in self.config.views.values()
                if view.source not in views
            }
        )
        if missing:
            raise ValueError(
                "DA3 input is missing configured source views: "
                f"{missing}; available={list(views)}"
            )
        prepared = [
            self._prepare_frame(views[view.source], view)
            for view in self.config.views.values()
        ]
        features = self.encoder(da3.prepare_images(prepared, self.config.image_size, self.device).unsqueeze(0))
        return {
            target: features[source].squeeze(0)
            for target, source in self.config.layer_map.items()
        }

    def _prepare_frame(
        self,
        frames: object,
        view: conditions.ConditionViewConfig,
    ) -> torch.Tensor:
        array = np.asarray(frames, dtype=np.uint8)
        if array.ndim == 4 and array.shape[-1] == 3:
            array = array[-1]
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                "DA3 camera input must have shape [H,W,3] or "
                f"[T,H,W,3], got {array.shape}"
            )
        source_height, source_width = array.shape[:2]
        x0, y0, x1, y1 = view.crop_bounds(source_height, source_width)
        array = array[y0:y1, x0:x1]
        frame = torch.from_numpy(array.copy()).permute(2, 0, 1)
        frame = frame.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        return frame / 255.0 * 2.0 - 1.0
