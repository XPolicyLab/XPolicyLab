"""Cosmos AR conditioned on episode-start and current images, both in UND only."""

import torch

from .cosmos3_fastwam_ar import Cosmos3FastWAMAR
from .mrope import build_und_position_ids


class Cosmos3FastWAMReferenceAR(Cosmos3FastWAMAR):
    """Two-image UND without new parameters or changes to the causal VAE stream."""

    def _prepare_ar_und(self, *, images, prompts, reference_image=None):
        """UND layout: [start | reference | end | start | current | end | text].

        Each image has its own spatial mRoPE grid. Reference input is [B,3,H,W];
        current images are [B*K,3,H,W]. Encode the episode reference once per
        sample, then reuse its vision features across the K training chunks.
        """
        if reference_image is None:
            raise ValueError("Cosmos3FastWAMReferenceAR requires the episode's reference_image.")
        reference_image = reference_image.to(self.device, self.torch_dtype)
        if reference_image.ndim != 4 or reference_image.shape[1:] != images.shape[1:]:
            raise ValueError("reference_image must be [B,3,H,W] with the current image layout.")
        chunks, remainder = divmod(images.shape[0], reference_image.shape[0])
        if remainder:
            raise ValueError("Current-image batch must contain K boundaries per reference image.")
        reference_vision, grid = self._encode_und_image(reference_image)
        reference_vision = reference_vision.repeat_interleave(chunks, dim=0)
        current_vision, _ = self._encode_und_image(images)
        n_image = current_vision.shape[1]
        ids, pad, image_mask = self._build_und_inputs(prompts, n_image)
        # Reuse the pretrained image delimiters, placing the reference before current.
        ids = torch.cat([ids[:, :n_image + 2], ids], dim=1)
        pad = torch.cat([torch.zeros_like(pad[:, :n_image + 2]), pad], dim=1)
        image_mask = torch.cat([image_mask[:, :n_image + 2], image_mask], dim=1)
        vision = torch.cat([reference_vision, current_vision], dim=1)
        tokens = self.backbone._embed_und_tokens(ids, vision, image_mask)
        reference_positions, reference_end = build_und_position_ids(
            text_len=0, n_img=n_image, image_grid_thw=grid,
            spatial_merge_size=self.vision_merge_size, device=self.device,
        )
        current_positions, current_end = build_und_position_ids(
            text_len=self.tokenizer_max_len, n_img=n_image, image_grid_thw=grid,
            spatial_merge_size=self.vision_merge_size, device=self.device,
        )
        positions = torch.cat([reference_positions, current_positions + reference_end], dim=1)
        return tokens, pad, positions, int(reference_end + current_end)

    def warmup_inference(self, **kwargs):
        height, width = kwargs["video_size"]
        reference = torch.zeros(1, 3, height, width, device=self.device, dtype=self.torch_dtype)
        return super().warmup_inference(**kwargs, reference_image=reference)
