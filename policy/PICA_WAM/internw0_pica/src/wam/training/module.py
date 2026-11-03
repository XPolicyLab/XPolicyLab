import torch.nn as nn

from ..inference.frame_joint import infer_frame_joint_window
from ..model.modules.codecs import video_latent_codec as video_codec
from .losses import frame_training_loss


class WAMTrainingModule(nn.Module):
    """DDP/DeepSpeed-facing objective wrapper around the core InternW0-delta model."""

    def __init__(self, wam: nn.Module) -> None:
        super().__init__()
        self.wam = wam

    def forward(
        self,
        sample=None,
        *,
        operation: str = "training_loss",
        tiled: bool = False,
        **kwargs,
    ):
        if operation == "training_loss":
            return frame_training_loss(
                self.wam, sample, tiled=tiled, global_step=kwargs.pop("global_step", None)
            )
        if operation == "encode_video_latents":
            return video_codec.encode_video_latents(self.wam, **kwargs)
        if operation == "decode_latents":
            return video_codec.decode_latents(self.wam, **kwargs)
        if operation == "infer_frame_joint_window":
            return infer_frame_joint_window(self.wam, **kwargs)
        raise ValueError(f"Unsupported WAM distributed operation: {operation!r}")
