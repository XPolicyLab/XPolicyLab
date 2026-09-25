from __future__ import annotations

import ast
import inspect
import re
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

DEFAULT_UNDERSTANDING_PROMPT = (
    "Given the current multi-camera observation and task instruction, "
    "represent the task-relevant visual state needed to choose the next action."
)


def _profile_section(profiler, name: str):
    return profiler.section(name) if profiler is not None else nullcontext()


def resolve_hf_snapshot(model_path: str | Path) -> str:
    path = Path(str(model_path)).expanduser()
    refs_main = path / "refs" / "main"
    snapshots = path / "snapshots"
    if refs_main.is_file() and snapshots.is_dir():
        revision = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if candidate.exists():
            return str(candidate)
    if snapshots.is_dir():
        candidates = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if candidates:
            return str(candidates[-1])
    return str(path)


@contextmanager
def _disabled_hf_progress_bars():
    """Temporarily hide HuggingFace/Transformers progress bars during VLM load."""
    transformers_enabled = None
    hub_disabled = None
    try:
        from transformers.utils import logging as transformers_logging

        transformers_enabled = transformers_logging.is_progress_bar_enabled()
        transformers_logging.disable_progress_bar()
    except Exception:
        transformers_logging = None
    try:
        from huggingface_hub import utils as hub_utils

        hub_disabled = hub_utils.are_progress_bars_disabled()
        hub_utils.disable_progress_bars()
    except Exception:
        hub_utils = None
    try:
        yield
    finally:
        if transformers_logging is not None and transformers_enabled:
            transformers_logging.enable_progress_bar()
        if hub_utils is not None and hub_disabled is False:
            hub_utils.enable_progress_bars()


@contextmanager
def _patched_fla_triton_source_lookup():
    """Patch Python 3.11 inspect only while importing FLA-backed Qwen kernels."""
    original_getsourcelines = inspect.getsourcelines

    def patched_getsourcelines(obj):
        lines, start_line = original_getsourcelines(obj)
        code = getattr(obj, "__code__", None)
        filename = str(getattr(code, "co_filename", ""))
        if "/fla/" not in filename.replace("\\", "/"):
            return lines, start_line
        src = "".join(lines)
        if re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE):
            return lines, start_line

        path = Path(filename)
        try:
            file_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            tree = ast.parse("".join(file_lines))
        except Exception:
            return lines, start_line

        name = getattr(obj, "__name__", None)
        first_line = max(1, int(getattr(code, "co_firstlineno", start_line)))
        candidates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == name
            and int(getattr(node, "lineno", 0)) >= first_line
            and getattr(node, "end_lineno", None) is not None
        ]
        if not candidates:
            return lines, start_line
        node = min(candidates, key=lambda item: int(item.lineno))
        fixed_lines = file_lines[int(node.lineno) - 1 : int(node.end_lineno)]
        return fixed_lines, int(node.lineno)

    inspect.getsourcelines = patched_getsourcelines
    try:
        yield
    finally:
        inspect.getsourcelines = original_getsourcelines


def _as_list(value: Any, batch_size: int, default: str = "") -> list[str]:
    if value is None:
        return [default] * batch_size
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, Sequence):
        out = [str(v) for v in value]
        if len(out) == batch_size:
            return out
        if len(out) == 1:
            return out * batch_size
    return [default] * batch_size


def _install_vectorized_token_validation(processor: Any) -> bool:
    """Patch HF multimodal token-count validation with a batched tensor fast path."""
    original_check = getattr(processor, "_check_special_mm_tokens", None)
    if original_check is None or not callable(original_check):
        return False

    def _check_special_mm_tokens(self, text: list[str], text_inputs: Any, modalities: list[str]):
        input_ids = None
        if isinstance(text_inputs, dict):
            input_ids = text_inputs.get("input_ids")
        else:
            try:
                input_ids = text_inputs["input_ids"]
            except Exception:
                input_ids = getattr(text_inputs, "input_ids", None)

        if torch.is_tensor(input_ids) and input_ids.ndim == 2:
            for modality in modalities:
                token_str = getattr(self, f"{modality}_token", None)
                token_id = getattr(self, f"{modality}_token_id", None)
                if token_str is not None and token_id is not None:
                    ids_count = input_ids.eq(token_id).sum(dim=-1).tolist()
                    text_count = [sample.count(token_str) for sample in text]

                    if ids_count != text_count:
                        raise ValueError(
                            f"Mismatch in `{modality}` token count between text and `input_ids`. Got ids={ids_count} and text={text_count}. "
                            "Likely due to `truncation='max_length'`. Please disable truncation or increase `max_length`."
                        )
            return

        return original_check(text, text_inputs, modalities)

    processor._check_special_mm_tokens = MethodType(_check_special_mm_tokens, processor)
    return True


def _view_names_from_string(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split("|") if part]
    if isinstance(value, Sequence):
        return [str(part) for part in value]
    return []


def _pretty_view_name(name: str) -> str:
    text = str(name).strip()
    key = text.lower().replace("-", "_")
    key = key.split("/")[-1].split(".")[-1]
    if "left" in key and ("wrist" in key or "hand" in key):
        return "left wrist camera"
    if "right" in key and ("wrist" in key or "hand" in key):
        return "right wrist camera"
    if "wrist" in key or key in {"hand", "hand_rgb"}:
        return "wrist camera"
    if "head" in key or "high" in key or "base" in key:
        return "head/front camera"
    if key in {"image", "rgb", "main"}:
        return "main camera"
    return key.replace("_", " ") or text


def _pretty_view_names(names: Sequence[str], views_per_frame: int) -> list[str]:
    out = [_pretty_view_name(name) for name in names[: max(int(views_per_frame), 0)]]
    if out:
        return out
    return [f"camera view {idx + 1}" for idx in range(max(int(views_per_frame), 1))]


def _clean_task_prompt(prompt: str) -> str:
    text = str(prompt or "").strip()
    marker = "executing the following instruction:"
    lower = text.lower()
    if marker in lower:
        start = lower.index(marker) + len(marker)
        return text[start:].strip().rstrip(".")
    return text


class FrozenQwenVLExtractor(nn.Module):
    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        understanding_prompt: Optional[str] = None,
        train_vlm: bool = False,
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = True,
        vlm_batch_size: int = 0,
        precompute_metadata: bool = True,
        max_pixels: int = 65536,
        prompt_cache_size: int = 256,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        resolved_path = resolve_hf_snapshot(model_path)
        with _disabled_hf_progress_bars(), _patched_fla_triton_source_lookup():
            self.processor = AutoProcessor.from_pretrained(
                resolved_path,
                trust_remote_code=trust_remote_code,
                max_pixels=int(max_pixels),
            )
            self.processor_token_validation_vectorized = _install_vectorized_token_validation(
                self.processor
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                resolved_path,
                dtype=dtype,
                device_map=None,
                trust_remote_code=trust_remote_code,
            )
        self.model.to(device=device, dtype=dtype)
        self.train_vlm = bool(train_vlm)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if not hasattr(self.model, "model"):
            raise TypeError(
                f"Unsupported VLM class {type(self.model).__name__}: expected a `.model` backbone "
                "that returns `last_hidden_state`."
            )
        self.set_trainable(self.train_vlm)
        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.hidden_size = int(text_config.hidden_size)
        self.dtype = dtype
        self.understanding_prompt = str(understanding_prompt or DEFAULT_UNDERSTANDING_PROMPT)
        # Zero means one full VLM batch. A positive value is an explicit
        # memory-saving microbatch cap.
        self.vlm_batch_size = max(0, int(vlm_batch_size))
        self.precompute_metadata = bool(precompute_metadata)
        self.prompt_cache_size = max(0, int(prompt_cache_size))
        self._formatted_prompt_cache: OrderedDict[
            tuple[Any, ...], str
        ] = OrderedDict()
        self.prompt_cache_hits = 0
        self.prompt_cache_misses = 0
        self._vision_metadata_cache: OrderedDict[
            tuple[Any, ...], dict[str, torch.Tensor]
        ] = OrderedDict()
        self.vision_metadata_cache_hits = 0
        self.vision_metadata_cache_misses = 0

    def set_trainable(self, train_vlm: Optional[bool] = None) -> None:
        if train_vlm is not None:
            self.train_vlm = bool(train_vlm)
        if self.train_vlm:
            self.train()
            self.model.requires_grad_(True)
        else:
            self.eval()
            self.model.requires_grad_(False)

    def _format_prompt(
        self,
        prompt: str,
        num_images: int,
        *,
        frame_labels: Sequence[str] | None = None,
        views_per_frame: int = 1,
        view_names: Sequence[str] | None = None,
    ) -> str:
        labels = list(frame_labels or ["current"])
        views_per_frame = max(int(views_per_frame), 1)
        view_names = list(view_names or [])
        temporal_lines = []
        for frame_idx, label in enumerate(labels):
            raw_label = str(label)
            if raw_label == "current":
                pretty_label = "Current observation"
            else:
                pretty_label = raw_label
            view_desc = ", ".join(_pretty_view_names(view_names, views_per_frame))
            temporal_lines.append(f"- Frame {frame_idx + 1}: {pretty_label}; views={view_desc}.")
        temporal_context = "\n".join(temporal_lines)
        task_text = _clean_task_prompt(prompt)
        instruction = (
            f"Task: {task_text}\n"
            "Observations:\n"
            f"{temporal_context}\n"
            f"Input Images: {max(int(num_images), 1)}; ordered by temporal frame, then camera view.\n"
            f"Output: {self.understanding_prompt}"
        )
        if not hasattr(self.processor, "apply_chat_template"):
            return instruction
        content = [{"type": "image"} for _ in range(max(int(num_images), 1))]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _format_prompt_cached(
        self,
        prompt: str,
        num_images: int,
        *,
        frame_labels: Sequence[str] | None = None,
        views_per_frame: int = 1,
        view_names: Sequence[str] | None = None,
    ) -> str:
        capacity = max(0, int(getattr(self, "prompt_cache_size", 0)))
        if capacity == 0:
            return self._format_prompt(
                prompt,
                num_images,
                frame_labels=frame_labels,
                views_per_frame=views_per_frame,
                view_names=view_names,
            )
        labels = tuple(str(value) for value in (frame_labels or ("current",)))
        names = tuple(str(value) for value in (view_names or ()))
        key = (
            str(prompt),
            max(int(num_images), 1),
            labels,
            max(int(views_per_frame), 1),
            names,
        )
        cache = self._formatted_prompt_cache
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            self.prompt_cache_hits += 1
            return cached
        formatted = self._format_prompt(
            prompt,
            num_images,
            frame_labels=labels,
            views_per_frame=views_per_frame,
            view_names=names,
        )
        cache[key] = formatted
        cache.move_to_end(key)
        while len(cache) > capacity:
            cache.popitem(last=False)
        self.prompt_cache_misses += 1
        return formatted

    def _move_inputs(
        self,
        inputs: dict[str, Any],
        *,
        keep_cpu_keys: set[str] | frozenset[str] = frozenset(),
        preserve_float_dtype_keys: set[str] | frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if key in keep_cpu_keys:
                    moved[key] = value.to(device="cpu")
                elif value.is_floating_point() and key not in preserve_float_dtype_keys:
                    moved[key] = value.to(device=self.device, dtype=self.dtype)
                else:
                    moved[key] = value.to(device=self.device)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _cpu_tensor(value: Any, *, name: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"VLM processor `{name}` must be a tensor, got {type(value)}.")
        return value.detach().to(device="cpu")

    def _vision_metadata_key(self, image_grid_thw: torch.Tensor) -> tuple[Any, ...]:
        flat_grid = tuple(int(value) for value in image_grid_thw.reshape(-1).tolist())
        return (
            tuple(int(value) for value in image_grid_thw.shape),
            flat_grid,
            self.device.type,
            self.device.index,
        )

    def _cached_vision_metadata(
        self,
        image_grid_thw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Precompute grid-only Qwen vision tensors once per device and grid.

        Bilinear weights are deliberately constructed on the target device so
        the cached fast path matches the checkpoint's original GPU metadata
        construction.  ``cu_seqlens`` remains on CPU for the current SDPA
        implementation: every vision layer calls ``lengths.tolist()``, and a
        CPU tensor avoids a device synchronization without changing split
        sizes or attention math.
        """
        key = self._vision_metadata_key(image_grid_thw)
        cached = self._vision_metadata_cache.get(key)
        if cached is not None:
            self._vision_metadata_cache.move_to_end(key)
            self.vision_metadata_cache_hits += 1
            return cached

        from transformers.vision_utils import (
            get_vision_bilinear_indices_and_weights,
            get_vision_cu_seqlens,
            get_vision_position_ids,
        )

        backbone = self.model.model
        visual = getattr(backbone, "visual", None)
        if visual is None or not hasattr(visual, "num_grid_per_side"):
            raise TypeError(
                "VLM metadata precompute requires a Qwen vision model with "
                "`num_grid_per_side`."
            )
        spatial_merge_size = int(visual.config.spatial_merge_size)
        grid_on_device = image_grid_thw.to(device=self.device)
        with torch.no_grad():
            bilinear_indices, bilinear_weights = (
                get_vision_bilinear_indices_and_weights(
                    grid_on_device,
                    num_grid_per_side=int(visual.num_grid_per_side),
                    spatial_merge_size=spatial_merge_size,
                )
            )
            vision_position_ids = get_vision_position_ids(
                grid_on_device,
                spatial_merge_size=spatial_merge_size,
            )
        cu_seqlens = get_vision_cu_seqlens(image_grid_thw)
        attention_impl = str(
            getattr(visual.config, "_attn_implementation", "") or ""
        )
        if attention_impl.startswith("flash_attention"):
            cu_seqlens = cu_seqlens.to(device=self.device)

        cached = {
            "image_bilinear_indices": bilinear_indices,
            "image_bilinear_weights": bilinear_weights,
            "image_position_ids": vision_position_ids,
            "image_cu_seqlens": cu_seqlens,
        }
        self._vision_metadata_cache[key] = cached
        self._vision_metadata_cache.move_to_end(key)
        while len(self._vision_metadata_cache) > 16:
            self._vision_metadata_cache.popitem(last=False)
        self.vision_metadata_cache_misses += 1
        return cached

    def _prepare_model_metadata(
        self,
        inputs: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[torch.Tensor]]:
        """Build data-dependent Qwen metadata before moving token tensors.

        Qwen's default multimodal forward converts GPU tensors to Python lists
        while constructing M-RoPE and vision metadata.  The processor already
        returns the relevant integer tensors on CPU, so construct language
        positions there and pass Transformers' supported precomputed vision
        kwargs.  Image pixels, token IDs, model weights, and attention math are
        unchanged.
        """
        if not bool(getattr(self, "precompute_metadata", False)):
            return inputs, None
        required = ("input_ids", "mm_token_type_ids", "image_grid_thw")
        if any(name not in inputs for name in required):
            return inputs, None
        if inputs.get("pixel_values_videos") is not None:
            # The WAM online path is image-only. Preserve the upstream path for
            # callers that later add video inputs rather than partially
            # precomputing mixed image/video metadata.
            return inputs, None

        prepared = dict(inputs)
        input_ids = self._cpu_tensor(prepared["input_ids"], name="input_ids")
        mm_token_type_ids = self._cpu_tensor(
            prepared["mm_token_type_ids"], name="mm_token_type_ids"
        )
        image_grid_thw = self._cpu_tensor(
            prepared["image_grid_thw"], name="image_grid_thw"
        )
        attention_mask_value = prepared.get("attention_mask")
        attention_mask = (
            self._cpu_tensor(attention_mask_value, name="attention_mask")
            if attention_mask_value is not None
            else None
        )

        with torch.no_grad():
            language_position_ids, rope_deltas = self.model.model.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        prepared["position_ids"] = language_position_ids
        # Keep the grid on CPU: all GPU vision metadata is supplied below, and
        # Qwen's post-vision split_sizes.tolist() then becomes synchronization-free.
        prepared["image_grid_thw"] = image_grid_thw
        prepared.update(self._cached_vision_metadata(image_grid_thw))
        return prepared, rope_deltas

    def forward(
        self,
        frames: torch.Tensor | Sequence[torch.Tensor],
        prompts: Sequence[str],
        *,
        view_names: Sequence[str] | None = None,
        frame_labels: Sequence[Sequence[str]] | Sequence[str] | None = None,
        view_valid_mask: Optional[torch.Tensor] = None,
        profiler=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames_are_batched = isinstance(frames, torch.Tensor)
        if frames_are_batched:
            if frames.ndim != 6:
                raise ValueError(
                    f"`frames` must be [B,T,V,C,H,W], got {tuple(frames.shape)}"
                )
            frame_batch = list(frames.unbind(0))
        elif isinstance(frames, Sequence):
            frame_batch = list(frames)
        else:
            raise TypeError(
                "`frames` must be a tensor or sequence of [T,V,C,H,W] tensors, "
                f"got {type(frames)}"
            )
        if not frame_batch:
            raise ValueError("`frames` must contain at least one sample.")
        for sample_frames in frame_batch:
            if not isinstance(sample_frames, torch.Tensor) or sample_frames.ndim != 5:
                raise ValueError(
                    "Each `frames` sample must be [T,V,C,H,W], "
                    f"got {type(sample_frames)} with shape={getattr(sample_frames, 'shape', None)}"
                )
            if int(sample_frames.shape[0]) <= 0 or int(sample_frames.shape[1]) <= 0:
                raise ValueError(
                    f"Each `frames` sample needs at least one frame and view, got {tuple(sample_frames.shape)}"
                )
            if int(sample_frames.shape[2]) != 3:
                raise ValueError(
                    f"Each `frames` sample must have RGB channels, got {tuple(sample_frames.shape)}"
                )
        with _profile_section(profiler, "image_cast_rescale"):
            if frames_are_batched:
                # One allocation/kernel for the common fixed-resolution batch
                # instead of one float32 allocation and kernel per sample.
                normalized_frames = frames.detach().to(dtype=torch.float32).div_(255.0)
                frame_batch = list(normalized_frames.unbind(0))
            else:
                frame_batch = [
                    sample_frames.detach().to(dtype=torch.float32).div_(255.0)
                    for sample_frames in frame_batch
                ]
        batch_size = len(frame_batch)
        temporal_lengths = [int(sample_frames.shape[0]) for sample_frames in frame_batch]
        if view_valid_mask is None:
            view_valid_batch = [
                torch.ones(
                    sample_frames.shape[:2],
                    dtype=torch.bool,
                    device="cpu",
                )
                for sample_frames in frame_batch
            ]
        else:
            view_valid_mask = torch.as_tensor(
                view_valid_mask, device="cpu", dtype=torch.bool
            )
            if view_valid_mask.ndim != 3 or int(
                view_valid_mask.shape[0]
            ) != batch_size:
                raise ValueError(
                    "`view_valid_mask` must be [B,T,V], got "
                    f"{tuple(view_valid_mask.shape)}."
                )
            view_valid_batch = []
            for index, sample_frames in enumerate(frame_batch):
                expected = tuple(sample_frames.shape[:2])
                actual = tuple(view_valid_mask[index].shape)
                if actual != expected:
                    raise ValueError(
                        "`view_valid_mask` must match each [T,V] frame grid, "
                        f"sample={index} got={actual} expected={expected}."
                    )
                sample_valid = view_valid_mask[index]
                if not bool(sample_valid.any().item()):
                    raise ValueError(
                        f"VLM sample {index} contains no valid camera views."
                    )
                view_valid_batch.append(sample_valid)
        prompt_list = _as_list(prompts, batch_size, default="")
        view_name_list = _as_list(view_names, batch_size, default="")
        if frame_labels is None:
            label_list = [
                ["current"] if temporal_len == 1 else [f"frame_{i + 1}" for i in range(temporal_len)]
                for temporal_len in temporal_lengths
            ]
        elif isinstance(frame_labels, str):
            label_list = [
                [str(frame_labels)] * temporal_len
                for temporal_len in temporal_lengths
            ]
        elif len(frame_labels) > 0 and isinstance(frame_labels[0], str):  # type: ignore[index]
            labels = [str(item) for item in frame_labels]  # type: ignore[assignment]
            if any(len(labels) != temporal_len for temporal_len in temporal_lengths):
                raise ValueError(
                    "`frame_labels` length must match every sample's temporal frames: "
                    f"{len(labels)} vs {temporal_lengths}"
                )
            label_list = [labels for _ in range(batch_size)]
        else:
            label_list = [[str(label) for label in labels] for labels in frame_labels]  # type: ignore[union-attr]
            if len(label_list) != batch_size:
                raise ValueError(f"`frame_labels` batch length must be {batch_size}, got {len(label_list)}")
            for labels, temporal_len in zip(label_list, temporal_lengths):
                if len(labels) != temporal_len:
                    raise ValueError(
                        f"Each `frame_labels` item must have {temporal_len} labels, got {len(labels)}"
                    )

        def _run_vlm_batch(texts: list[str], image_groups: list[list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
            images = [image for group in image_groups for image in group]
            with _profile_section(profiler, "processor"):
                inputs = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    images_kwargs={
                        "do_rescale": False,
                        "input_data_format": "channels_first",
                        "device": self.device,
                    },
                )
            with _profile_section(profiler, "model_metadata_precompute"):
                inputs, rope_deltas = self._prepare_model_metadata(dict(inputs))

            keep_cpu_keys: set[str] = set()
            preserve_float_dtype_keys: set[str] = set()
            if rope_deltas is not None:
                keep_cpu_keys.add("image_grid_thw")
                image_cu_seqlens = inputs.get("image_cu_seqlens")
                if (
                    torch.is_tensor(image_cu_seqlens)
                    and image_cu_seqlens.device.type == "cpu"
                ):
                    keep_cpu_keys.add("image_cu_seqlens")
                # Qwen constructs interpolation weights in float32 even for a
                # BF16 model.  Preserve that exact dtype on the cached path.
                preserve_float_dtype_keys.add("image_bilinear_weights")
            with _profile_section(profiler, "model_input_h2d"):
                inputs = self._move_inputs(
                    inputs,
                    keep_cpu_keys=keep_cpu_keys,
                    preserve_float_dtype_keys=preserve_float_dtype_keys,
                )
                if rope_deltas is not None:
                    # Supplying position_ids skips Qwen's default M-RoPE
                    # construction, so mirror its only model-state side effect.
                    self.model.model.rope_deltas = rope_deltas.to(
                        device=self.device
                    )
            grad_context = torch.enable_grad() if self.train_vlm else torch.no_grad()
            with _profile_section(profiler, "backbone"):
                with grad_context:
                    outputs = self.model.model(
                        **inputs,
                        return_dict=True,
                        use_cache=False,
                    )
                    hidden = outputs.last_hidden_state
            with _profile_section(profiler, "postprocess"):
                if hidden.ndim == 2:
                    if len(texts) != 1:
                        raise RuntimeError(f"VLM hidden state lost batch dim for batch={len(texts)}: {tuple(hidden.shape)}")
                    hidden = hidden.unsqueeze(0)
                if hidden.ndim != 3:
                    raise RuntimeError(f"VLM hidden state must be [B,S,H], got {tuple(hidden.shape)}.")
                if int(hidden.shape[0]) != len(texts):
                    raise RuntimeError(f"VLM hidden batch mismatch: {int(hidden.shape[0])} vs {len(texts)}.")
                hidden = hidden.to(dtype=self.dtype)
                mask = inputs.get("attention_mask")
                if mask is None:
                    mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                else:
                    mask = mask.to(device=hidden.device, dtype=torch.bool)
                    if mask.ndim == 1:
                        mask = mask.unsqueeze(0)
                    if mask.ndim != 2:
                        raise RuntimeError(f"VLM attention mask must be [B,S], got {tuple(mask.shape)}.")
                    if tuple(mask.shape) != tuple(hidden.shape[:2]):
                        raise RuntimeError(
                            "VLM attention mask must exactly match hidden states, "
                            f"got mask={tuple(mask.shape)} hidden={tuple(hidden.shape[:2])}."
                        )
            return hidden, mask

        def _run_vlm_microbatches(texts: list[str], image_groups: list[list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
            if len(texts) != len(image_groups):
                raise RuntimeError(f"VLM text/image batch mismatch: {len(texts)} vs {len(image_groups)}")
            if self.vlm_batch_size <= 0 or len(texts) <= self.vlm_batch_size:
                return _run_vlm_batch(texts, image_groups)
            hidden_chunks: list[torch.Tensor] = []
            mask_chunks: list[torch.Tensor] = []
            for start in range(0, len(texts), self.vlm_batch_size):
                end = min(start + self.vlm_batch_size, len(texts))
                hidden_i, mask_i = _run_vlm_batch(texts[start:end], image_groups[start:end])
                hidden_chunks.append(hidden_i)
                mask_chunks.append(mask_i)
            if len(hidden_chunks) == 1:
                return hidden_chunks[0], mask_chunks[0]
            max_len = max(int(hidden.shape[1]) for hidden in hidden_chunks)
            padded_hidden: list[torch.Tensor] = []
            padded_mask: list[torch.Tensor] = []
            for hidden, mask in zip(hidden_chunks, mask_chunks):
                pad_len = max_len - int(hidden.shape[1])
                if pad_len > 0:
                    hidden = torch.cat(
                        [
                            hidden,
                            hidden.new_zeros((int(hidden.shape[0]), pad_len, int(hidden.shape[2]))),
                        ],
                        dim=1,
                    )
                    mask = torch.cat(
                        [
                            mask,
                            torch.zeros((int(mask.shape[0]), pad_len), dtype=torch.bool, device=mask.device),
                        ],
                        dim=1,
                    )
                padded_hidden.append(hidden)
                padded_mask.append(mask)
            return torch.cat(padded_hidden, dim=0), torch.cat(padded_mask, dim=0)

        texts: list[str] = []
        image_groups: list[list[torch.Tensor]] = []
        with _profile_section(profiler, "prompt_template"):
            for idx in range(batch_size):
                names = _view_names_from_string(view_name_list[idx])
                images: list[torch.Tensor] = []
                sample_frames = frame_batch[idx]
                sample_view_valid = view_valid_batch[idx]
                num_temporal_frames = int(sample_frames.shape[0])
                views_per_frame = max(
                    int(sample_view_valid[frame_idx].sum().item())
                    for frame_idx in range(num_temporal_frames)
                )
                for frame_idx in range(num_temporal_frames):
                    images.extend(
                        image
                        for view_idx, image in enumerate(
                            sample_frames[frame_idx].unbind(0)
                        )
                        if bool(sample_view_valid[frame_idx, view_idx])
                    )
                text = self._format_prompt_cached(
                    prompt_list[idx],
                    len(images),
                    frame_labels=label_list[idx],
                    views_per_frame=views_per_frame,
                    view_names=names,
                )
                texts.append(text)
                image_groups.append(images)
        return _run_vlm_microbatches(texts, image_groups)


class QwenVLUnderstandingEncoder(nn.Module):
    def __init__(
        self,
        *,
        vlm_model_path: str,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prompt: Optional[str] = None,
        train_vlm: bool = False,
        vlm_gradient_checkpointing: bool = False,
        save_vlm_weights: Optional[bool] = None,
        trust_remote_code: bool = True,
        vlm_batch_size: int = 0,
        precompute_metadata: bool = True,
        max_pixels: int = 65536,
        prompt_cache_size: int = 256,
        use_view_valid_mask: bool = True,
    ) -> None:
        super().__init__()
        self.train_vlm = bool(train_vlm)
        self.use_view_valid_mask = bool(use_view_valid_mask)
        self.save_vlm_weights = bool(self.train_vlm if save_vlm_weights is None else save_vlm_weights)
        self.vlm = FrozenQwenVLExtractor(
            vlm_model_path,
            device=device,
            dtype=dtype,
            understanding_prompt=prompt,
            train_vlm=self.train_vlm,
            gradient_checkpointing=vlm_gradient_checkpointing,
            trust_remote_code=trust_remote_code,
            vlm_batch_size=vlm_batch_size,
            precompute_metadata=precompute_metadata,
            max_pixels=max_pixels,
            prompt_cache_size=prompt_cache_size,
        )
        self.context_dim = int(self.vlm.hidden_size)
        self.to(device=device, dtype=dtype)
        self.set_trainable()

    def set_trainable(self) -> None:
        self.vlm.set_trainable(self.train_vlm)

    def trainable_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("vlm.") and not self.train_vlm:
                continue
            if param.requires_grad:
                yield param

    def adapter_state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.train_vlm and self.save_vlm_weights:
            state["vlm_model"] = self.vlm.model.state_dict()
        return state

    def load_adapter_state_dict(self, state: dict[str, Any], strict: bool = False) -> None:
        if "vlm_model" in state:
            self.vlm.model.load_state_dict(state["vlm_model"], strict=strict)

    def forward(
        self,
        frames: torch.Tensor | Sequence[torch.Tensor],
        prompts: Sequence[str],
        *,
        view_names: Sequence[str] | None = None,
        frame_labels: Sequence[Sequence[str]] | Sequence[str] | None = None,
        valid_mask: Optional[torch.Tensor] = None,
        view_valid_mask: Optional[torch.Tensor] = None,
        profiler=None,
    ) -> dict[str, torch.Tensor]:
        full_batch_size = (
            int(frames.shape[0])
            if isinstance(frames, torch.Tensor)
            else len(frames)
        )
        valid_indices = None
        if valid_mask is not None:
            valid_mask = valid_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
            if int(valid_mask.numel()) != full_batch_size:
                raise ValueError(
                    "`valid_mask` must have one value per VLM input: "
                    f"got {tuple(valid_mask.shape)} for batch={full_batch_size}."
                )
            if not bool(valid_mask.any().item()):
                raise ValueError("VLM understanding received no valid refresh inputs.")
            if not bool(valid_mask.all().item()):
                valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
                index_list = valid_indices.tolist()
                if isinstance(frames, torch.Tensor):
                    frame_indices = valid_indices.to(device=frames.device)
                    frames = frames.index_select(0, frame_indices)
                else:
                    frames = [frames[index] for index in index_list]
                prompts = [prompts[index] for index in index_list]
                if view_names is not None:
                    view_names = [view_names[index] for index in index_list]
                if view_valid_mask is not None:
                    view_indices = valid_indices.to(
                        device=view_valid_mask.device
                    )
                    view_valid_mask = view_valid_mask.index_select(
                        0, view_indices
                    )
                if (
                    frame_labels is not None
                    and len(frame_labels) == full_batch_size
                    and len(frame_labels) > 0
                    and not isinstance(frame_labels[0], str)
                ):
                    frame_labels = [frame_labels[index] for index in index_list]
        vlm_hidden, vlm_mask = self.vlm(
            frames,
            prompts,
            view_names=view_names,
            frame_labels=frame_labels,
            view_valid_mask=(
                view_valid_mask if self.use_view_valid_mask else None
            ),
            profiler=profiler,
        )
        if vlm_hidden.ndim != 3:
            raise ValueError(f"`vlm_hidden` must be [B,S,H], got {tuple(vlm_hidden.shape)}")
        vlm_context = vlm_hidden
        if valid_indices is not None and int(valid_indices.numel()) < full_batch_size:
            compact_context = vlm_context
            compact_mask = vlm_mask.to(
                device=compact_context.device, dtype=torch.bool
            )
            vlm_context = compact_context.new_zeros(
                full_batch_size,
                int(compact_context.shape[1]),
                int(compact_context.shape[2]),
            )
            vlm_mask = torch.zeros(
                (full_batch_size, int(compact_mask.shape[1])),
                dtype=torch.bool,
                device=compact_context.device,
            )
            scatter_indices = valid_indices.to(device=compact_context.device)
            vlm_context.index_copy_(0, scatter_indices, compact_context)
            vlm_mask.index_copy_(0, scatter_indices, compact_mask)
        return {
            "vlm_context": vlm_context,
            "vlm_mask": vlm_mask.to(device=vlm_context.device, dtype=torch.bool),
        }
