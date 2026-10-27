"""Serving adapter for :class:`MachEmbodiedUnifiedModel`.

Maps the sim server payload ``(images, instruction, state)`` to MachEmbodiedUnifiedModel's
``predict_actions`` and returns ``(T, raw_action_dim)`` env-ready actions.

Normalization reuses each map-style dataset's field statistics and transforms.
LIBERO uses native LeRobot v2.1 min/max; RoboDojo uses the H48 delta-joint
codec and q01/q99 statistics. Both use the shared training prompt renderer.

Matches the ``PolicyServer`` contract: ``predict_action(images, instruction,
state, dataset_name=None, action_horizon=None, **ignored) -> (T, Da) np.float32``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF

from leap.data.world_unified.metadata import Metadata, render_instruction

logger = logging.getLogger(__name__)

_SUPPORTED_MOSAIC_LAYOUTS = {"horizontal", "pyramid"}


class MachEmbodiedUnifiedPolicy:
    def __init__(
        self,
        model,
        tokenizer=None,
        per_view_size: int | Sequence[int] = 256,
        mosaic_layout: str = "horizontal",
        raw_action_dim: int = 7,
        num_inference_steps: Optional[int] = None,
        domain_id: int = 0,
        max_text_len: int = 512,
        device: str = "cuda",
        normalizers: Optional[Dict[str, Any]] = None,  # {dataset_name: training normalizer}
        action_codecs: Optional[Dict[str, Any]] = None,
        domain_ids: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Metadata]] = None,
        video_out_dir: Optional[str] = None,   # if set, save one generated future video per task
        quantize_images: Optional[bool] = None,
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
        assert self.tokenizer is not None, "MachEmbodiedUnifiedPolicy needs a tokenizer"
        self.mosaic_layout = str(mosaic_layout).lower()
        if self.mosaic_layout not in _SUPPORTED_MOSAIC_LAYOUTS:
            raise ValueError(
                "mosaic_layout must be one of "
                f"{sorted(_SUPPORTED_MOSAIC_LAYOUTS)}, got {mosaic_layout!r}"
            )
        if isinstance(per_view_size, bool):
            raise TypeError("per_view_size must be an integer or [height, width]")
        if isinstance(per_view_size, int):
            if per_view_size <= 0:
                raise ValueError("per_view_size must be positive")
            self.per_view = int(per_view_size)
            self.per_view_shape = (self.per_view, self.per_view)
        else:
            values = tuple(per_view_size)
            if len(values) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in values
            ):
                raise ValueError(
                    "per_view_size sequence must be two positive integers [height, width]"
                )
            self.per_view_shape = (int(values[0]), int(values[1]))
            self.per_view = self.per_view_shape
        if self.mosaic_layout == "horizontal" and not isinstance(self.per_view, int):
            raise ValueError(
                "rectangular per_view_size requires mosaic_layout='pyramid'"
            )
        if self.mosaic_layout == "pyramid" and isinstance(self.per_view, int):
            raise ValueError(
                "pyramid mosaic requires per_view_size=[height, width] from the config"
            )
        if self.mosaic_layout == "pyramid" and any(
            value % 2 for value in self.per_view_shape
        ):
            raise ValueError(
                "pyramid per_view_size height and width must both be even, got "
                f"{self.per_view_shape}"
            )
        # Preserve existing defaults; the RoboDojo entry point enables this
        # for both layouts to match its LeRobot provider. LIBERO keeps floats.
        self.quantize_images = (
            self.mosaic_layout == "pyramid"
            if quantize_images is None else bool(quantize_images)
        )
        self.raw_action_dim = raw_action_dim
        self.num_inference_steps = num_inference_steps
        self.domain_id = domain_id
        self.max_text_len = max_text_len
        self.device = device
        self.state_dim = int(model.config.state_dim)
        self.normalizers = normalizers or {}
        self.action_codecs = action_codecs or {}
        self.domain_ids = domain_ids or {}
        self.metadata = metadata or {}
        self.video_out_dir = video_out_dir
        self._video_seen: set = set()   # (dataset_name, instruction) already saved
        if video_out_dir:
            import os
            os.makedirs(video_out_dir, exist_ok=True)

    def _resolve_normalizer(self, dataset_name: Optional[str]):
        if not self.normalizers:
            return None
        if dataset_name and dataset_name in self.normalizers:
            return self.normalizers[dataset_name]
        if len(self.normalizers) == 1:
            return next(iter(self.normalizers.values()))
        logger.warning("dataset_name=%r not in normalizers %s; no (de)normalization applied.",
                       dataset_name, list(self.normalizers.keys()))
        return None

    def _resolve_metadata(self, dataset_name: Optional[str]) -> Optional[Metadata]:
        if not self.metadata:
            return None
        if dataset_name and dataset_name in self.metadata:
            return self.metadata[dataset_name]
        if len(self.metadata) == 1:
            return next(iter(self.metadata.values()))
        logger.warning(
            "dataset_name=%r not in metadata %s; control metadata omitted.",
            dataset_name,
            list(self.metadata.keys()),
        )
        return None

    def _resolve_action_codec(self, dataset_name: Optional[str]):
        """Resolve a map-style ME-U0 state/action codec.

        Map-style LeRobot datasets do not use ``NormalizeRobotData``.  They
        normalize named fields with ``LinearNormalizer`` and then concatenate
        them in ``shape_meta`` order.  A codec is therefore mutually exclusive
        with the legacy world-unified normalizer for one request.
        """
        if not self.action_codecs:
            return None
        if dataset_name and dataset_name in self.action_codecs:
            return self.action_codecs[dataset_name]
        if len(self.action_codecs) == 1:
            return next(iter(self.action_codecs.values()))
        logger.warning(
            "dataset_name=%r not in action_codecs %s; map-style normalization disabled.",
            dataset_name,
            list(self.action_codecs.keys()),
        )
        return None

    def _build_mosaic(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        if self.mosaic_layout == "pyramid":
            if len(images) not in {2, 3}:
                raise ValueError(
                    "pyramid mosaic requires two or three cameras ordered as "
                    f"[head, left_wrist, right_wrist], got {len(images)}"
                )
            height, width = self.per_view_shape
            views = []
            for image in images:
                view = (
                    torch.from_numpy(np.ascontiguousarray(image))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
                # Match the LeRobot provider: resize each raw view, store it
                # as uint8, then let the processor convert it back to float.
                # Wrist downsampling happens after this shared source resize.
                view = TF.resize(view, [height, width], antialias=True)
                if self.quantize_images:
                    view = view.mul(255).to(torch.uint8).float().div(255)
                views.append(view)
            head = TF.resize(views[0], [height, width], antialias=True)
            left_wrist = TF.resize(
                views[1], [height // 2, width // 2], antialias=True
            )
            right_wrist = (
                TF.resize(views[2], [height // 2, width // 2], antialias=True)
                if len(views) == 3
                else torch.zeros_like(left_wrist)
            )
            wrists = torch.cat([left_wrist, right_wrist], dim=-1)
            mosaic = torch.cat([head, wrists], dim=-2) * 2.0 - 1.0
            return mosaic.unsqueeze(0)

        # LIBERO keeps float resize; RoboDojo matches its uint8 provider.
        views = []
        for img in images:
            t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            t = TF.resize(t, [self.per_view, self.per_view], antialias=True)
            if self.quantize_images:
                t = t.mul(255).to(torch.uint8).float().div(255)
            views.append(t)
        mosaic = torch.cat(views, dim=-1) * 2.0 - 1.0     # (3,256,512*nviews)
        return mosaic.unsqueeze(0)                        # (1,3,256,W)

    @torch.no_grad()
    def predict_action(
        self,
        images: Sequence[np.ndarray],
        instruction: str,
        state: Optional[np.ndarray] = None,
        dataset_name: Optional[str] = None,
        action_horizon: Optional[int] = None,
        **_ignored: Any,
    ) -> np.ndarray:
        codec = self._resolve_action_codec(dataset_name)
        norm = None if codec is not None else self._resolve_normalizer(dataset_name)
        metadata = self._resolve_metadata(dataset_name)
        did = int(self.domain_ids.get(dataset_name, self.domain_id)) if self.domain_ids else self.domain_id
        if codec is not None:
            raw_a = int(codec.canonical_action_dim)
            raw_s = int(codec.canonical_state_dim)
        else:
            raw_a = int(norm.config.action_dim) if norm is not None else self.raw_action_dim
            raw_s = int(norm.config.state_dim) if norm is not None else self.state_dim

        video = self._build_mosaic(images).to(self.device)
        main_image = torch.from_numpy(np.ascontiguousarray(images[0])).permute(2, 0, 1).float() / 255.0
        main_image = TF.resize(main_image, [224, 224], antialias=True)
        if self.quantize_images:
            # The training provider quantizes the independently resized main
            # camera with round(), unlike its diffusion-camera truncation.
            main_image = main_image.clamp(0, 1).mul(255).round().to(torch.uint8).float().div(255)
        main_image = main_image.to(self.device)

        # state: normalize raw dims (train↔infer aligned), then pad to state_dim
        if state is None:
            state_t = torch.zeros(self.state_dim, dtype=torch.float32)
        else:
            if codec is not None:
                s = codec.encode_state(state).squeeze(0)
            else:
                s = torch.as_tensor(np.asarray(state), dtype=torch.float32).flatten()
            if norm is not None:
                s = norm.apply({"states": s[:raw_s]})["states"]
            if s.shape[0] < self.state_dim:
                s = torch.cat([s, torch.zeros(self.state_dim - s.shape[0])])
            state_t = s[: self.state_dim]
        state_t = state_t.to(self.device)

        from leap.data.world_unified.instruction_template import build_instruction_text_ids

        text_ids = build_instruction_text_ids(
            self.tokenizer,
            instruction,
            max_text_len=self.max_text_len,
            metadata=metadata,
        ).to(self.device)
        model_instruction = render_instruction(instruction, metadata)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            key = (str(dataset_name), str(instruction))
            want_video = bool(self.video_out_dir) and key not in self._video_seen
            if want_video:
                self._video_seen.add(key)
                act, rgb = self.model.predict_actions(
                    video=video, state=state_t, text_ids=text_ids,
                    domain_id=did, num_steps=self.num_inference_steps, return_video=True,
                    main_image=main_image,
                    instruction=model_instruction, raw_action_dim=raw_a, raw_state_dim=raw_s,
                )
                try:
                    self._save_future_video(rgb, dataset_name, instruction)
                except Exception as e:  # noqa: BLE001 — visualization must never break eval
                    logger.warning("save future video failed: %s", e)
            else:
                act = self.model.predict_actions(
                    video=video, state=state_t, text_ids=text_ids,
                    domain_id=did, num_steps=self.num_inference_steps,
                    main_image=main_image,
                    instruction=model_instruction, raw_action_dim=raw_a, raw_state_dim=raw_s,
                )
        act = act.float().cpu()                           # normalized canonical model output
        if codec is not None:
            # Delta-joint checkpoints predict actions relative to the same
            # current native state that conditioned the model.
            return codec.decode_action(act, native_state=state).astype(np.float32)
        act = act[:, :raw_a]
        if norm is not None:                              # denormalize -> raw env units
            act = norm.unapply({"actions": act})["actions"]
        return act.float().numpy().astype(np.float32)

    def _save_future_video(self, rgb, dataset_name, instruction) -> None:
        """Save one model-generated future clip. rgb: (3,T,H,W) in ~[-1,1]."""
        import os
        import re
        import imageio  # optional dep; only reached when video_out_dir is set
        thwc = (rgb.clamp(-1, 1).add(1).div(2).mul(255).round()
                .to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy())  # (T,H,W,3)
        tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{dataset_name}_{instruction}")[:80]
        path = os.path.join(self.video_out_dir, f"gen_{tag}.mp4")
        imageio.mimsave(path, list(thwc), fps=8, format="mp4")
        logger.info("[ME_U0] saved generated future video: %s (%d frames)", path, thwc.shape[0])
