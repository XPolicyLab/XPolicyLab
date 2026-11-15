"""Shared instruction-to-prompt transformations for embodied models."""

from __future__ import annotations

import pydoc
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_ACTION_LABELS = {
    "qpos_left": "left arm joints",
    "endpose_left": "left EEF pose",
    "gripper_left": "left gripper",
    "qpos_right": "right arm joints",
    "endpose_right": "right EEF pose",
    "gripper_right": "right gripper",
    "qpos_torso": "torso joints",
    "endpose_torso": "torso EEF pose",
    "qpos_head": "head joints",
    "chassis": "chassis",
}
_ACTION_LABELS_ZH = {
    "qpos_left": "左臂关节",
    "endpose_left": "左末端执行器位姿",
    "gripper_left": "左夹爪",
    "qpos_right": "右臂关节",
    "endpose_right": "右末端执行器位姿",
    "gripper_right": "右夹爪",
    "qpos_torso": "躯干关节",
    "endpose_torso": "躯干末端执行器位姿",
    "qpos_head": "头部关节",
    "chassis": "底盘",
}
_ACTION_DIM = 55
_ACTION_SLOTS = (
    ("qpos_left", 0, 6),
    ("endpose_left", 7, 15),
    ("gripper_left", 16, 16),
    ("qpos_right", 17, 23),
    ("endpose_right", 24, 32),
    ("gripper_right", 33, 33),
    ("qpos_torso", 34, 40),
    ("endpose_torso", 41, 49),
    ("qpos_head", 50, 51),
    ("chassis", 52, 54),
)
_STITCH_LAYOUT_TEMPLATES = {
    "top_high_bottom_left_right": (
        "{0} on top, {1} at bottom-left, {2} at bottom-right"
    ),
}
_STITCH_LAYOUT_TEMPLATES_ZH = {
    "top_high_bottom_left_right": (
        "{0}位于上方，{1}位于左下方，{2}位于右下方"
    ),
}
_VIEW_NAMES_ZH = {
    "head": "头部",
    "high": "高位",
    "left_wrist": "左腕",
    "right_wrist": "右腕",
}
_SUPPORTED_LANGUAGES = ("en", "zh")


def _caption(sample: Mapping[str, Any]) -> str:
    caption = sample.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Prompt builder requires a non-empty string caption")
    return caption.strip()


def _label(identifier: str) -> str:
    return identifier.replace("_", " ").strip().title()


def _validate_language(lang: str) -> str:
    if lang not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported prompt language {lang!r}; "
            f"expected one of {_SUPPORTED_LANGUAGES}"
        )
    return lang


def _view_label(identifier: str, lang: str) -> str:
    name = identifier.removeprefix("cam_").removesuffix("_camera")
    if lang == "zh":
        return f"{_VIEW_NAMES_ZH.get(name, name.replace('_', ' '))}视角"
    return f"{name.replace('_', ' ').strip()} view"


def _single_view_label(identifier: str, lang: str) -> str:
    parts = identifier.removeprefix("cam_").removesuffix("_camera").split("_")
    if "head" in parts:
        parts = ["head"]
    if lang == "zh":
        name = "_".join(parts)
        return f"单一{_VIEW_NAMES_ZH.get(name, ' '.join(parts))}视角"
    return f"single {' '.join(parts)} view"


def _format_range(start: int, end: int) -> str:
    return f"[{start}]" if start == end else f"[{start}-{end}]"


def _contiguous_ranges(indices: Sequence[int]) -> tuple[tuple[int, int], ...]:
    start = end = indices[0]
    ranges: list[tuple[int, int]] = []
    for index in indices[1:]:
        if index == end + 1:
            end = index
            continue
        ranges.append((start, end))
        start = end = index
    ranges.append((start, end))
    return tuple(ranges)


def _format_action_component(
    name: str,
    start: int,
    end: int,
    active_ranges: tuple[tuple[int, int], ...],
    lang: str,
) -> str:
    slot = _format_range(start, end)
    if active_ranges == ((start, end),):
        return f"{name} {slot}"
    active = ("，" if lang == "zh" else ", ").join(
        _format_range(range_start, range_end)
        for range_start, range_end in active_ranges
    )
    if lang == "zh":
        return f"{name} {slot}（激活 {active}）"
    return f"{name} {slot} (active {active})"


class PromptBuilder(ABC):
    """Transform one instruction-bearing sample into model prompt text."""

    @abstractmethod
    def build(self, sample: Mapping[str, Any]) -> str:
        """Build prompt text from one sample."""

        raise NotImplementedError


class CaptionPromptBuilder(PromptBuilder):
    """Use the original caption as the complete prompt."""

    def build(self, sample: Mapping[str, Any]) -> str:
        """Return the stripped sample caption."""

        return _caption(sample)


class EmbodiedPromptBuilder(PromptBuilder):
    """Render instruction, embodiment, video layout, and action slots.

    Args:
        embodiment: Optional fixed embodiment name. The sample value is used
            when this is unset.
        lang: Default template language, either ``"en"`` or ``"zh"``.
    """

    def __init__(
        self,
        embodiment: str | None = None,
        lang: str = "en",
    ) -> None:
        self._embodiment = embodiment
        self._lang = _validate_language(lang)

    def build(
        self,
        sample: Mapping[str, Any],
        *,
        lang: str | None = None,
    ) -> str:
        """Build an embodied prompt, optionally overriding its language."""

        language = self._lang if lang is None else _validate_language(lang)
        embodiment = self._embodiment or str(sample.get("embodiment", "unknown"))
        action = ("，" if language == "zh" else ", ").join(
            self._resolve_action_components(sample, language)
        )
        video_context = self._resolve_video_context(sample, language)
        if language == "zh":
            sections = [
                f"指令：{_caption(sample)}",
                f"本体：{_label(embodiment)}。",
                (
                    f"视频布局：{video_context['video_layout']}。"
                    if "video_layout" in video_context
                    else None
                ),
                (
                    f"视频：{video_context['video_view']}。"
                    if "video_view" in video_context
                    else None
                ),
                f"动作：{_ACTION_DIM}维；{action}。其余槽位均为填充。",
            ]
            return "\n".join(
                section for section in sections if section is not None
            )
        sections = [
            f"Instruction: {_caption(sample)}",
            f"Embodiment: {_label(embodiment)}.",
            (
                f"Video layout: {video_context['video_layout']}."
                if "video_layout" in video_context
                else None
            ),
            (
                f"Video: {video_context['video_view']}."
                if "video_view" in video_context
                else None
            ),
            f"Action: {_ACTION_DIM}D; {action}. All other slots are padding.",
        ]
        return "\n".join(section for section in sections if section is not None)

    def _resolve_action_components(
        self,
        sample: Mapping[str, Any],
        lang: str,
    ) -> list[str]:
        action_mask = sample.get("action_mask")
        if action_mask is None:
            raise ValueError("Embodied prompt builder requires action_mask")
        mask = torch.as_tensor(action_mask, dtype=torch.bool)
        if mask.numel() % _ACTION_DIM != 0:
            raise ValueError(
                "canonical55 action mask must contain a multiple of 55 values, "
                f"got shape {tuple(mask.shape)}"
            )
        active = mask.reshape(-1, _ACTION_DIM).any(dim=0).tolist()
        components: list[str] = []
        for key, start, end in _ACTION_SLOTS:
            active_indices = [
                index
                for index in range(start, end + 1)
                if active[index]
            ]
            if active_indices:
                components.append(
                    _format_action_component(
                        (
                            _ACTION_LABELS_ZH[key]
                            if lang == "zh"
                            else _ACTION_LABELS[key]
                        ),
                        start,
                        end,
                        _contiguous_ranges(active_indices),
                        lang,
                    )
                )
        if not components:
            raise ValueError("canonical55 action mask has no active slots")
        return components

    def _resolve_video_context(
        self,
        sample: Mapping[str, Any],
        lang: str,
    ) -> dict[str, str]:
        selected_views = sample.get("video")
        meta = sample.get("meta")
        if not isinstance(selected_views, Mapping) or not isinstance(meta, Mapping):
            return {}
        resolved_config = meta.get("resolved_config")
        if not isinstance(resolved_config, Mapping):
            return {}
        videos = resolved_config.get("videos")
        if not isinstance(videos, Sequence):
            return {}
        for video in videos:
            if not isinstance(video, Mapping) or video.get("name") not in selected_views:
                continue
            sources = video.get("sources")
            if not isinstance(sources, Sequence) or isinstance(sources, str) or not sources:
                continue
            source_names = [str(source) for source in sources]
            if len(source_names) == 1:
                return {
                    "video_view": _single_view_label(source_names[0], lang)
                }
            transforms = video.get("transforms")
            if not isinstance(transforms, Sequence):
                continue
            stitch = next(
                (
                    item
                    for item in transforms
                    if isinstance(item, Mapping)
                    and item.get("config_type") == "Stitch"
                ),
                None,
            )
            templates = (
                _STITCH_LAYOUT_TEMPLATES_ZH
                if lang == "zh"
                else _STITCH_LAYOUT_TEMPLATES
            )
            template = (
                templates.get(str(stitch.get("layout")))
                if stitch is not None
                else None
            )
            if template is not None:
                return {
                    "video_layout": template.format(
                        *[
                            _view_label(source, lang)
                            for source in source_names
                        ]
                    )
                }
        return {}


def build_prompt_builder(
    config: Mapping[str, Any] | PromptBuilder | None = None,
) -> PromptBuilder:
    """Construct a prompt builder; caption-only is the default."""

    if isinstance(config, PromptBuilder):
        return config
    builder_config = dict(config or {})
    builder_type = builder_config.pop("type", CaptionPromptBuilder)
    if isinstance(builder_type, str):
        builder_type = pydoc.locate(builder_type)
    if not isinstance(builder_type, type) or not issubclass(
        builder_type,
        PromptBuilder,
    ):
        raise TypeError(
            "Prompt builder type must inherit PromptBuilder, "
            f"got {builder_type!r}"
        )
    return builder_type(**builder_config)
