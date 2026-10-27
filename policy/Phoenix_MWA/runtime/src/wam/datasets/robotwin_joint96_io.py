"""Shared text and clean RGB semantics for online and cached RoboTwin Joint96."""
import torch


def official_text_padding(context, valid):
    """Upstream T5 convention: zero padded features; keep ALL keys visible.

    Never mutate a shared/read-only cache row. All-valid is deliberately not
    equivalent to masking out the padding tokens in attention.
    """
    if context.shape != (128, 4096) or valid.shape != (128,):
        raise ValueError('Invalid T5 cache row geometry')
    result = context.masked_fill(~valid.bool()[:, None], 0)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite valid T5 features')
    return result, torch.ones_like(valid, dtype=torch.bool)


class FramewiseResize:
    """torchvision resize accepts NCHW, not the Joint96 V,T,C,H,W layout."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, video):
        if video.ndim != 5:
            return self.transform(video)
        views, frames = video.shape[:2]
        result = self.transform(video.reshape(-1, *video.shape[-3:]))
        return result.reshape(views, frames, *result.shape[-3:])


class BoundedJoint96RGB:
    """Apply upstream transforms, then bound normalized RGB before either AE.

    Also used by the offline extractor with augmentation disabled. Cached
    training overrides _visual_inputs and never executes this RGB path.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resize_transform = FramewiseResize(self.resize_transform)

    def _format_video_window(self, *args, **kwargs):
        video = super()._format_video_window(*args, **kwargs)
        # Bicubic interpolation can overshoot even when all input pixels are
        # in [0,1]. Keep the encoder's strict range guard; repair at the source.
        if not torch.isfinite(video).all():
            raise ValueError('Nonfinite RGB after preprocessing')
        return video.clamp(-1.0, 1.0)
