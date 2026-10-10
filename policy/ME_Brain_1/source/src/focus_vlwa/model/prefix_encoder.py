"""Visual and language prefix encoding for Focus-VLWA."""

import math

import torch
from torch import Tensor

from focus_vlwa.model.head_history import encode_head_history


class VisualPrefixEncoder:
    def _embed_siglip(self, img):
        return self._apply_checkpoint(lambda x: self.joint_experts.embed_image(x), img)

    def _to_nchw(self, img: Tensor) -> Tensor:
        if img.ndim == 4 and img.shape[-1] == 3:
            return img.permute(0, 3, 1, 2)
        return img

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, observation=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed full head history, current camera views, and language as one prefix."""
        embs = []
        pad_masks = []
        att_masks = []

        history_images = getattr(observation, "history_images", None) if observation is not None else None
        if history_images is None:
            raise ValueError("Head-history models require explicit history images and a validity mask")
        cam_embs, cam_pads, n_tok = self._embed_prefix_with_head_history(images, img_masks, observation)
        embs.extend(cam_embs)
        pad_masks.extend(cam_pads)
        att_masks += [0] * n_tok
        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.joint_experts.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def _embed_prefix_with_head_history(self, images, img_masks, observation):
        """Prepend 20 head-history frames x 16 tokens to full current-camera tokens."""
        current_embs = [self._embed_siglip(self._to_nchw(img)) for img in images]
        current_pads = [
            mask.to(dtype=torch.bool)[:, None].expand(emb.shape[:2])
            for mask, emb in zip(img_masks, current_embs, strict=True)
        ]
        history_embs, history_pad = encode_head_history(
            self._embed_siglip, observation.history_images, observation.history_mask, current_embs[0]
        )
        embs = [history_embs, *current_embs]
        pads = [history_pad, *current_pads]
        return embs, pads, sum(emb.shape[1] for emb in embs)
