# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from transformers import (
    AutoImageProcessor,
    BartTokenizerFast,
    ProcessorMixin,
    Qwen2TokenizerFast,
)
from typing import List, Union, Dict, Any, Optional
from collections import OrderedDict
import torch

from .image_preprocessing import (
    IMAGE_RESIZE_LETTERBOX,
    QWEN_POLICY_PAD_RGB,
    letterbox_image,
    normalize_image_resize_mode,
)

DEFAULT_ACTION_LANGUAGE_MAX_LENGTH = 50


def _config_value(config_or_dict, name: str, default=None):
    if isinstance(config_or_dict, dict):
        return config_or_dict.get(name, default)
    return getattr(config_or_dict, name, default)


def apply_action_language_max_length(processor, config_or_dict) -> int:
    """Apply the checkpoint action-language contract to a processor."""
    cap = int(
        _config_value(
            config_or_dict,
            "action_language_max_length",
            DEFAULT_ACTION_LANGUAGE_MAX_LENGTH,
        )
        or DEFAULT_ACTION_LANGUAGE_MAX_LENGTH
    )
    if cap <= 0:
        raise ValueError(f"action_language_max_length must be positive, got {cap}.")
    previous = int(
        getattr(processor, "language_max_length", DEFAULT_ACTION_LANGUAGE_MAX_LENGTH)
        or DEFAULT_ACTION_LANGUAGE_MAX_LENGTH
    )
    processor.language_max_length = cap
    if cap != previous and hasattr(processor, "clear_language_cache"):
        processor.clear_language_cache()
    return cap


class XVLAProcessor(ProcessorMixin):
    """
    XVLAProcessor: Unified multimodal processor for XVLA models.

    Handles:
      - Multi-view image inputs (e.g., from multiple cameras).
      - Batch processing for multiple samples.
      - Joint tokenization and image tensor preparation.

    This processor combines an image processor and a tokenizer under a single interface
    so that users can call it directly like:

        >>> processor = XVLAProcessor.from_pretrained("path/to/xvla")
        >>> inputs = processor(images=batch_images, language_instruction=batch_texts)

    It is fully compatible with the Hugging Face AutoProcessor API.

    Attributes
    ----------
    num_views : int, default=3
        Expected number of image views per sample. Missing views will be padded with zeros.
    language_max_length : int, default=50
        Maximum token length for text encoding.
    attributes : list
        Required by ProcessorMixin to know which submodules are stored and reloaded.
    image_processor_class : str
        The name of the associated image processor class.
    tokenizer_class : tuple(str)
        The names of compatible tokenizer classes.
    """

    num_views: int = 3
    main_camera_only: bool | None = None
    language_max_length: int = DEFAULT_ACTION_LANGUAGE_MAX_LENGTH

    # Hugging Face ProcessorMixin-required metadata
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    @classmethod
    def _get_arguments_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        processor_dict=None,
        **kwargs,
    ):
        component_kwargs = dict(kwargs)
        for key in (
            "enable_language_cache",
            "language_cache_size",
            "language_max_length",
            "num_views",
            "main_camera_only",
            "vlm_backend",
            "image_resize_mode",
            "image_pad_rgb",
        ):
            component_kwargs.pop(key, None)
        if processor_dict is None:
            processor_dict, _ = cls.get_processor_dict(
                pretrained_model_name_or_path,
                **component_kwargs,
            )
        vlm_backend = str(processor_dict.get("vlm_backend", "florence2")).strip().lower()
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        if vlm_backend == "qwen3_5":
            tokenizer = Qwen2TokenizerFast.from_pretrained(
                pretrained_model_name_or_path,
                **component_kwargs,
            )
        elif vlm_backend == "dm05":
            from transformers import GemmaTokenizer
            tokenizer = GemmaTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                **component_kwargs,
            )
        else:
            tokenizer = BartTokenizerFast.from_pretrained(
                pretrained_model_name_or_path,
                **component_kwargs,
            )
        return [image_processor, tokenizer]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        vlm_backend: str = "florence2",
        enable_language_cache: bool = True,
        language_cache_size: int = 8192,
        language_max_length: int = DEFAULT_ACTION_LANGUAGE_MAX_LENGTH,
        num_views: int = 3,
        main_camera_only: bool | None = None,
        image_resize_mode: str = "auto",
        image_pad_rgb: List[int] | tuple[int, int, int] = QWEN_POLICY_PAD_RGB,
    ):
        """
        Initialize XVLAProcessor.

        Parameters
        ----------
        image_processor : PreTrainedImageProcessor, optional
            The image processor used to normalize/resize images.
        tokenizer : PreTrainedTokenizer, optional
            The tokenizer used for text tokenization.
        num_views : int, default 3
            Number of camera-view slots the model was trained with.  This value
            is written into ``processor_config.json`` at checkpoint-save time
            so that the deploy side can auto-detect the correct slot count
            without the caller having to pass ``--num-views`` explicitly.

            Background: ``num_views`` controls how many slots
            ``XVLAProcessor.__call__`` pads the image batch to.  A checkpoint
            trained with ``--num_views 1`` expects a single-slot tensor; loading
            it with the default ``num_views=3`` injects 2×N extra
            ``aux_visual_proj(0)`` bias tokens the model never saw during
            training, which degrades step-0 MAE by ~22%.  Persisting the value
            here removes that silent train/deploy mismatch.

            Existing checkpoints that were saved before this field existed will
            load with the class default (3), which is the historical behaviour
            and is correct for every checkpoint trained with ``--num_views 3``
            or the V=3 head-only (num_views=3, mask=[1,0,0]) convention.

            NOTE: the field lands in ``processor_config.json`` (which is what
            ``ProcessorMixin.to_dict()`` writes), not ``preprocessor_config.json``
            -- the latter belongs to the image processor.
        main_camera_only : bool or None, default None
            Whether the checkpoint was trained with only the primary camera
            decoded (``XVLA_MAIN_CAMERA_ONLY=1``).  Persisted alongside
            ``num_views`` so the deploy side can auto-detect how many REAL
            cameras to feed.

            Why this is needed IN ADDITION to ``num_views``: the two are
            orthogonal, and ``num_views`` alone cannot distinguish the two
            configurations that matter most::

                num_views=3, main_camera_only=True   -> head-only, mask=[1,0,0]
                                                        (arms A-C, 07-30, 08-06)
                num_views=3, main_camera_only=False  -> three REAL cameras,
                                                        mask=[1,1,1] (arms D/E,
                                                        07-24)

            Both write ``num_views: 3``.  Serving a three-camera checkpoint
            without feeding the wrist cameras is a SILENT failure -- the model
            simply receives zeros in slots 1/2 that it never saw during
            training.  Historically this was avoided only by the caller
            remembering to pass ``--all-views``, which is exactly the class of
            mistake that made a K=6 image-history checkpoint unevaluable.

            ``None`` means "not recorded" and preserves the historical
            behaviour: the deploy side keeps whatever ``all_views`` the caller
            asked for.  It is deliberately NOT defaulted to ``True``, because
            that would silently mislabel the pre-existing three-camera
            checkpoints (07-24, arm B) as head-only.
        """
        # ProcessorMixin automatically saves these under self.image_processor / self.tokenizer
        super().__init__(image_processor, tokenizer)
        self.vlm_backend = str(vlm_backend or "florence2").strip().lower()
        self.enable_language_cache = bool(enable_language_cache)
        self.language_cache_size = int(max(0, language_cache_size))
        self.language_max_length = int(language_max_length)
        self.num_views = int(num_views)
        self.main_camera_only = (
            None if main_camera_only is None else bool(main_camera_only)
        )
        self.image_resize_mode = normalize_image_resize_mode(
            image_resize_mode,
            vlm_backend=self.vlm_backend,
        )
        self.image_pad_rgb = [int(value) for value in image_pad_rgb]
        if len(self.image_pad_rgb) != 3 or any(value < 0 or value > 255 for value in self.image_pad_rgb):
            raise ValueError(f"image_pad_rgb must contain three uint8 values, got {image_pad_rgb!r}.")
        if self.language_max_length <= 0:
            raise ValueError(f"language_max_length must be positive, got {self.language_max_length}.")
        self._language_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def clear_language_cache(self) -> None:
        self._language_cache.clear()

    def _cache_get(self, text: str) -> torch.Tensor | None:
        if not self.enable_language_cache or self.language_cache_size <= 0:
            return None
        value = self._language_cache.get(text)
        if value is None:
            return None
        self._language_cache.move_to_end(text, last=True)
        return value

    def _cache_put(self, text: str, input_ids: torch.Tensor) -> None:
        if not self.enable_language_cache or self.language_cache_size <= 0:
            return
        ids = input_ids.detach().to("cpu", copy=True)
        self._language_cache[text] = ids
        self._language_cache.move_to_end(text, last=True)
        while len(self._language_cache) > self.language_cache_size:
            self._language_cache.popitem(last=False)

    # ================== LANGUAGE ENCODING ==================
    def encode_language(self, language_instruction: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize one or more language instructions.

        Parameters
        ----------
        language_instruction : str or List[str]
            A single instruction or a batch of instructions.

        Returns
        -------
        Dict[str, torch.Tensor]
            {
              "input_ids": tensor of shape [B, L]
            }
        """
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]

        texts = [str(x) for x in language_instruction]
        if not texts:
            return {"input_ids": torch.empty((0, self.language_max_length), dtype=torch.long)}

        cached: list[torch.Tensor | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        seen_miss: set[str] = set()

        for idx, text in enumerate(texts):
            ids = self._cache_get(text)
            if ids is not None:
                cached[idx] = ids
                continue
            if text not in seen_miss:
                miss_texts.append(text)
                seen_miss.add(text)
            miss_indices.append(idx)

        if miss_texts:
            inputs = self.tokenizer(
                miss_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=self.language_max_length,
                truncation=True,
            )
            miss_ids = inputs["input_ids"]
            for text, ids in zip(miss_texts, miss_ids):
                self._cache_put(text, ids)

        batch_ids: list[torch.Tensor] = []
        for text, ids in zip(texts, cached):
            if ids is None:
                ids = self._cache_get(text)
            if ids is None:
                # Defensive fallback (should be rare).
                fallback = self.tokenizer(
                    [text],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.language_max_length,
                    truncation=True,
                )["input_ids"][0]
                self._cache_put(text, fallback)
                ids = fallback
            batch_ids.append(ids)

        return {"input_ids": torch.stack(batch_ids, dim=0)}

    def encode_dm05_multimodal(
        self,
        language_instruction: Union[str, List[str]],
        state: torch.Tensor,
        *,
        num_images: int,
        state_dim: int = 20,
        control_mode: str = "end effector",
    ) -> Dict[str, torch.Tensor]:
        """Build the OpenDM/Gemma3 DM05 prefix fields.

        DM05 does not consume proprioception as a separate expert argument.
        Its native training transform serializes the normalized state into
        ``States:`` tokens inside the user prompt and places image soft-token
        spans at the image positions.  Keep this method DM05-specific so the
        legacy Hy text contract used by Florence/Qwen remains unchanged.
        """
        if self.vlm_backend != "dm05":
            raise ValueError("encode_dm05_multimodal is only valid for vlm_backend='dm05'.")
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]
        tasks = [str(value) for value in language_instruction]
        if state.ndim != 2 or state.shape[0] != len(tasks):
            raise ValueError(
                "DM05 state must be [B,D] and align with language batch, got "
                f"state={tuple(state.shape)} tasks={len(tasks)}"
            )
        if num_images <= 0:
            raise ValueError(f"DM05 requires at least one image, got {num_images}.")
        state_dim = int(state_dim)
        if state_dim <= 0 or state_dim > state.shape[-1]:
            raise ValueError(
                f"DM05 state_dim must be in [1, {state.shape[-1]}], got {state_dim}"
            )
        control_mode = str(control_mode).strip()
        if not control_mode:
            raise ValueError("DM05 control_mode must be non-empty.")

        tokenizer = self.tokenizer
        boi = str(getattr(tokenizer, "boi_token", "<start_of_image>"))
        eoi = str(getattr(tokenizer, "eoi_token", "<end_of_image>"))
        image_token = str(getattr(tokenizer, "image_token", "<image_soft_token>"))
        # Some Transformers image-processor versions expose this optional
        # field as ``None`` when the processor is constructed manually.
        image_seq_length = int(getattr(self.image_processor, "image_seq_length", None) or 256)
        full_image = "\n\n" + boi + image_token * image_seq_length + eoi + "\n\n"

        # OpenDM's action_to_bin_tokens expects normalized values in [-1, 1].
        # RoboDojo supplies z-score state.  ``tanh`` provides a monotonic,
        # bounded bridge (roughly mapping +/-2.3 standard deviations to the
        # OpenDM +/-1 quantile range) without changing dataset-side tensors.
        state_bins = torch.floor(
            (state[:, :state_dim].detach().float().cpu().tanh() + 1.0)
            * (255.0 / 2.0)
        ).to(torch.long)
        prompts: list[str] = []
        for task, bins in zip(tasks, state_bins, strict=True):
            content = [
                {
                    "type": "text",
                    "text": (
                        f"Control mode: {control_mode}\n"
                        "Overall speed: 0.5\n"
                        f"Task: {task}.\n"
                    ),
                }
            ]
            for image_index in range(num_images):
                content.append({"type": "text", "text": f"Camera {image_index} image: "})
                content.append({"type": "image"})
            content.append(
                {
                    "type": "text",
                    "text": "States: " + " ".join(str(int(value)) for value in bins.tolist()),
                }
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # Gemma3Processor expands each BOI placeholder to BOI + 256 image
            # soft tokens + EOI before tokenization.
            prompts.append(rendered.replace(boi, full_image))

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        image_token_id = int(getattr(tokenizer, "image_token_id"))
        token_type_ids = (input_ids == image_token_id).to(torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"].to(torch.long),
            "token_type_ids": token_type_ids,
        }

    # ================== IMAGE ENCODING ==================
    def _prepare_images(self, images: List) -> tuple[List, Dict[str, Any]]:
        resize_mode = getattr(self, "image_resize_mode", "stretch")
        if resize_mode != IMAGE_RESIZE_LETTERBOX:
            return list(images), {}
        size = getattr(self.image_processor, "size", None)
        if isinstance(size, dict):
            target = (int(size["height"]), int(size["width"]))
        elif getattr(size, "height", None) is not None and getattr(size, "width", None) is not None:
            target = (int(size.height), int(size.width))
        else:
            raise ValueError("Letterbox preprocessing requires image_processor.size as {'height', 'width'}.")
        prepared = [
            letterbox_image(
                image,
                size=target,
                fill=tuple(getattr(self, "image_pad_rgb", QWEN_POLICY_PAD_RGB)),
            )
            for image in images
        ]
        return prepared, {"do_resize": False, "do_center_crop": False}

    def encode_image(
        self,
        images: Union[List, List[List]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess one or more sets of multi-view images.

        Parameters
        ----------
        images : List, List[List], or List[List[List]]
            Single sample (no history):    [img1, img2, ...]
            Batch (no history):            [[img1a, img1b], [img2a, img2b, img2c], ...]
            Single sample with history:    [[view0_t0, view0_t1, ...], [view1_t0, ...], ...]
                                           i.e. a list of per-view time series;
                                           slot K-1 in each series is the current frame.
            Batch with history:            [[[view0_t0, ...], [view1_t0, ...]], ...]
            Each image may be a PIL.Image, NumPy array, or torch.Tensor.

        kwargs : dict
            Extra arguments passed to the underlying image processor
            (e.g., `do_resize=False`, `size=(224,224)`).

        Returns
        -------
        Dict[str, torch.Tensor]
            Without history (K==1):
                {
                  "image_input": tensor [B, num_views, C, H, W],
                  "image_mask":  tensor [B, num_views]   (bool)
                }
            With history (K>=2):
                {
                  "image_input": tensor [B, num_views, K, C, H, W],
                  "image_mask":  tensor [B, num_views, K] (bool)
                }
            The K axis matches the training-time layout emitted by
            `LeRobotGenericHandler` when `_img_history_size() > 1` (see
            `datasets/domain_handler/robodojo.py`).
        """
        # Normalize to batch form + detect whether the caller passed a
        # per-view time series (history mode) or a flat multi-view list.
        #
        # Layout table (types after normalization):
        #   flat single       [img, img]                       → wrap → [[img, img]]
        #   flat batch        [[img, img], [img, img]]         → no wrap
        #   history single    [[img, img], [img, img], ...]    → wrap → [[[img,img], ...]]
        #     (each inner list is a per-view time series K frames long)
        #   history batch     [[[img,img], ...], [[img,img], ...]] → no wrap
        first = images[0]
        history_mode = False
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
            # Batched history: outer list = batch, first sample is a list of
            # per-view sequences.
            history_mode = True
        elif isinstance(first, (list, tuple)) and first and not isinstance(first[0], (list, tuple)):
            # Ambiguous: could be
            #   (a) flat batch — each `first` element is a per-sample view list
            #       of PIL/ndarray objects, or
            #   (b) history single — `first` is a per-view time series (list of
            #       PIL/ndarray frames) and the caller forgot to wrap in a
            #       batch axis.
            #
            # Disambiguate by looking at the second element: for flat batch,
            # `images[1]` would also be a list. For history single, every
            # element of `images` is a per-view time series (all lists).
            # Both cases would step through the same branch here — the safe
            # semantics-preserving choice is (a): flat batch. To use history
            # mode, callers must always provide a batch axis, i.e. wrap in a
            # list one more time. This preserves backward compat with every
            # existing single-frame caller.
            history_mode = False
        else:
            # `first` is a single image (PIL / ndarray / tensor): flat single.
            images = [images]
            history_mode = False

        batch_imgs, batch_masks = [], []
        batch_hist_sizes: list[torch.Tensor] = []  # per-sample [V] K_v, populated only in history mode

        if not history_mode:
            for sample_imgs in images:
                prepared, resize_kwargs = self._prepare_images(list(sample_imgs))
                resize_kwargs = {**kwargs, **resize_kwargs}
                processed = self.image_processor(
                    prepared,
                    return_tensors="pt",
                    **resize_kwargs,
                )["pixel_values"]
                V_exist = processed.size(0)

                # Pad to self.num_views
                if V_exist < self.num_views:
                    processed = torch.cat(
                        [processed,
                         processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                        dim=0,
                    )

                # Mask: True for valid slots, False for padding
                image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=processed.device)
                image_mask[:V_exist] = True

                batch_imgs.append(processed)
                batch_masks.append(image_mask)

            image_input = torch.stack(batch_imgs, dim=0)  # [B, num_views, C, H, W]
            image_mask = torch.stack(batch_masks, dim=0)  # [B, num_views]

            return {"image_input": image_input, "image_mask": image_mask}

        # -------- History mode: per-view K-frame stacks --------
        # `images` is [B][V_exist][K] where K may differ across views; we treat
        # the max K in the batch as the canonical stack depth.
        max_K = 0
        for sample in images:
            for view_frames in sample:
                if isinstance(view_frames, (list, tuple)):
                    max_K = max(max_K, len(view_frames))
        if max_K == 0:
            raise ValueError("encode_image(history_mode): no frames provided in any view.")

        for sample_views in images:
            V_exist = len(sample_views)
            # Per-view: process the K frames as a batch, then transpose to [K, C, H, W].
            per_view_stacks: list[torch.Tensor] = []
            per_view_valid: list[list[bool]] = []
            per_view_kv: list[int] = []  # K_v per view, for path-D sparsification metadata
            for view_frames in sample_views:
                frames_list = list(view_frames)
                Kv = len(frames_list)
                if Kv == 0:
                    raise ValueError("encode_image(history_mode): empty per-view frame list.")
                prepared, resize_kwargs = self._prepare_images(frames_list)
                resize_kwargs = {**kwargs, **resize_kwargs}
                processed = self.image_processor(
                    prepared,
                    return_tensors="pt",
                    **resize_kwargs,
                )["pixel_values"]
                # `processed` is [Kv, C, H, W]. Pad Kv up to max_K by
                # LEFT-padding with zeros. The model side reads the paired
                # `image_hist_size` metadata (emitted below) and SKIPS these
                # zero slots on the DaViT forward — the K_v==1 view bypasses
                # the MEM video tower entirely. This is the "path D"
                # sparsification (docs/memory-decisions.md 2026-08-20) — zero
                # padding is safe because it never enters the tower, and it
                # keeps the dataloader-side pad cost as cheap as possible.
                if Kv < max_K:
                    pad = processed.new_zeros(max_K - Kv, *processed.shape[1:])
                    processed = torch.cat([pad, processed], dim=0)
                per_view_stacks.append(processed)  # [max_K, C, H, W]
                # Slot k=max_K-Kv..max_K-1 are real; slot 0..max_K-Kv-1 are pad.
                valid_k = [False] * (max_K - Kv) + [True] * Kv
                per_view_valid.append(valid_k)
                per_view_kv.append(Kv)

            # Pad views up to self.num_views with zero stacks.
            if V_exist < self.num_views:
                per_view_stacks.extend(
                    per_view_stacks[0].new_zeros(*per_view_stacks[0].shape)
                    for _ in range(self.num_views - V_exist)
                )
                per_view_valid.extend([False] * max_K for _ in range(self.num_views - V_exist))
                # Padding views: K_v = 1 (the whole view is zero anyway; the
                # model reads image_mask.any(dim=2) which is False for these,
                # so the K_v value is only for shape consistency).
                per_view_kv.extend([1] * (self.num_views - V_exist))

            sample_stack = torch.stack(per_view_stacks[: self.num_views], dim=0)  # [V, K, C, H, W]
            sample_mask = torch.tensor(per_view_valid[: self.num_views], dtype=torch.bool)  # [V, K]
            sample_hist_size = torch.tensor(
                per_view_kv[: self.num_views], dtype=torch.int32
            )
            batch_imgs.append(sample_stack)
            batch_masks.append(sample_mask)
            batch_hist_sizes.append(sample_hist_size)

        image_input = torch.stack(batch_imgs, dim=0)  # [B, V, K, C, H, W]
        image_mask = torch.stack(batch_masks, dim=0)  # [B, V, K]
        image_hist_size = torch.stack(batch_hist_sizes, dim=0)  # [B, V]
        return {
            "image_input": image_input,
            "image_mask": image_mask,
            "image_hist_size": image_hist_size,
        }

    def encode_image_sequence(
        self,
        image_sequences: Union[List[List[List]], List[List]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess one or more sets of temporal multi-view images.

        Parameters
        ----------
        image_sequences : List or List[List[List]]
            Single sample:
              [[frame1_view1, frame2_view1, ...], [frame1_view2, ...], ...]
            Batch:
              [
                [[...view1 frames...], [...view2 frames...]],
                [[...view1 frames...]],
              ]

        Returns
        -------
        Dict[str, torch.Tensor]
            {
              "image_seq_input": tensor [B, T, V, C, H, W],
              "image_mask_seq": tensor [B, T, V]
            }
        """
        if not image_sequences:
            return {
                "image_seq_input": torch.empty((0, 0, self.num_views, 3, 0, 0), dtype=torch.float32),
                "image_mask_seq": torch.empty((0, 0, self.num_views), dtype=torch.bool),
            }

        first = image_sequences[0]
        if not isinstance(first, (list, tuple)) or (first and not isinstance(first[0], (list, tuple))):
            image_sequences = [image_sequences]  # type: ignore[list-item]

        encoded_samples: List[List[torch.Tensor]] = []
        max_frames = 0
        max_height = 0
        max_width = 0
        channels = 3

        for sample_views in image_sequences:
            sample_encoded: List[torch.Tensor] = []
            for view_frames in list(sample_views)[: self.num_views]:
                if not view_frames:
                    continue
                prepared, resize_kwargs = self._prepare_images(list(view_frames))
                resize_kwargs = {**kwargs, **resize_kwargs}
                processed = self.image_processor(
                    prepared,
                    return_tensors="pt",
                    **resize_kwargs,
                )["pixel_values"]
                if processed.ndim != 4:
                    raise ValueError(
                        f"image sequence processor output must be [T, C, H, W], got {tuple(processed.shape)}"
                    )
                sample_encoded.append(processed)
                max_frames = max(max_frames, int(processed.shape[0]))
                channels = int(processed.shape[1])
                max_height = max(max_height, int(processed.shape[2]))
                max_width = max(max_width, int(processed.shape[3]))
            encoded_samples.append(sample_encoded)

        if max_frames <= 0:
            return {
                "image_seq_input": torch.empty((len(encoded_samples), 0, self.num_views, channels, max_height, max_width)),
                "image_mask_seq": torch.zeros((len(encoded_samples), 0, self.num_views), dtype=torch.bool),
            }

        batch_inputs: List[torch.Tensor] = []
        batch_masks: List[torch.Tensor] = []
        for sample_encoded in encoded_samples:
            sample_tensor = torch.zeros(
                (self.num_views, max_frames, channels, max_height, max_width),
                dtype=sample_encoded[0].dtype if sample_encoded else torch.float32,
            )
            sample_mask = torch.zeros((max_frames, self.num_views), dtype=torch.bool)
            for view_idx, clip in enumerate(sample_encoded[: self.num_views]):
                num_frames, _, height, width = clip.shape
                sample_tensor[view_idx, :num_frames, :, :height, :width] = clip
                sample_mask[:num_frames, view_idx] = True
            batch_inputs.append(sample_tensor.permute(1, 0, 2, 3, 4))
            batch_masks.append(sample_mask)

        return {
            "image_seq_input": torch.stack(batch_inputs, dim=0),
            "image_mask_seq": torch.stack(batch_masks, dim=0),
        }

    # ================== COMBINED CALL ==================
    def __call__(
        self,
        images: Optional[Union[List, List[List]]] = None,
        language_instruction: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Combine image and text encoding into a unified multimodal input.

        Parameters
        ----------
        images : List or List[List], optional
            Single-sample or batched multi-view images.
        language_instruction : str or List[str], optional
            Corresponding text instructions.
        kwargs : dict
            Extra args passed to image processor.

        Returns
        -------
        Dict[str, torch.Tensor]
            {
              "input_ids": [B, L], optional,
              "image_input": [B, num_views, C, H, W], optional,
              "image_mask": [B, num_views], optional
            }
        """
        outputs: Dict[str, Any] = {}

        # Encode language if provided
        if language_instruction is not None:
            outputs.update(self.encode_language(language_instruction))

        # Encode image if provided
        if images is not None:
            outputs.update(self.encode_image(images, **kwargs))

        # Sanity check for batch alignment
        if "input_ids" in outputs and "image_input" in outputs:
            assert outputs["input_ids"].size(0) == outputs["image_input"].size(0), (
                f"Batch mismatch: text batch {outputs['input_ids'].size(0)} "
                f"!= image batch {outputs['image_input'].size(0)}"
            )
        return outputs
