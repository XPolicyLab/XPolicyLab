"""The release trainer: Wan22Trainer plus the checks every recipe relies on."""

from __future__ import annotations

from prismwam.trainer import Wan22Trainer


class ReleaseTrainer(Wan22Trainer):
    """Require the shared RGB/depth backbone and a run that reaches ``max_steps``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        rgb, depth = (
            getattr(model, "rgb_expert", None),
            getattr(model, "video_expert", None),
        )
        if rgb is not None and depth is not None and rgb.blocks is not depth.blocks:
            raise ValueError("RGB and Depth must share the visual backbone.")

    def train(self):
        result = super().train()
        if self.global_step != self.max_steps:
            raise RuntimeError(
                f"Training stopped at {self.global_step}/{self.max_steps}."
            )
        return result
