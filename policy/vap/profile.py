"""Explicit train/serve contracts for VAP RoboDojo evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


LEGACY_PROFILE = "legacy_a40v20"
A32V16_320X256_PROFILE = "a32v16_320x256"
EE_V3_LFU_STATE_PROFILE = "a32v16_320x256_eev3_lfu_state"
EE_V3_LFU_STATE_MINMAX_PROFILE = "a32v16_320x256_eev3_lfu_state_minmax"
EE_V3_LFU_STATE_GLOBALNORM_PROFILE = "a32v16_320x256_eev3_lfu_state_globalnorm"
EE_V3_LFU_NOSTATE_PROFILE = "a32v16_320x256_eev3_lfu_nostate"
EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE = "a32v16_320x256_eev3_lfu_nostate_globalnorm"
EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE = (
    "a32v16_320x256_eev3_lfu_nostate_globalnorm_follower"
)
EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE = (
    "a32v16_320x256_eev3_lfu_nostate_globalnorm_memory"
)
EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE = (
    "a32v8_320x256_eev3_lfu_nostate_globalnorm_memory"
)
EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE = (
    "a64v16_320x256_eev3_lfu_nostate_globalnorm"
)
EE_V3_PROFILES = frozenset(
    {
        EE_V3_LFU_STATE_PROFILE,
        EE_V3_LFU_STATE_MINMAX_PROFILE,
        EE_V3_LFU_STATE_GLOBALNORM_PROFILE,
        EE_V3_LFU_NOSTATE_PROFILE,
        EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE,
        EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE,
        EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE,
        EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE,
        EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE,
    }
)

EE_V3_ACTION_KEYS = (
    ("master_left_ee_cartesian_pos_relative", 3),
    ("master_left_ee_rotation_6D_relative", 6),
    ("master_left_gripper", 1),
    ("master_right_ee_cartesian_pos_relative", 3),
    ("master_right_ee_rotation_6D_relative", 6),
    ("master_right_gripper", 1),
)
EE_V3_STATE_KEYS = (
    ("state_left_ee_cartesian_pos", 3),
    ("state_left_ee_rotation_6D", 6),
    ("state_left_gripper", 1),
    ("state_right_ee_cartesian_pos", 3),
    ("state_right_ee_rotation_6D", 6),
    ("state_right_gripper", 1),
)
EE_V3_FOLLOWER_ACTION_KEY_MAPPING = (
    ("master_left_ee_cartesian_pos_relative", "follow_left_position"),
    ("master_left_ee_rotation_6D_relative", "follow_left_rotation"),
    ("master_left_gripper", "follow_left_gripper"),
    ("master_right_ee_cartesian_pos_relative", "follow_right_position"),
    ("master_right_ee_rotation_6D_relative", "follow_right_rotation"),
    ("master_right_gripper", "follow_right_gripper"),
)


@dataclass(frozen=True)
class RoboDojoEvaluationProfile:
    """One indivisible image/action/video/normalizer serving contract."""

    name: str
    protocol_version: int
    action_steps: int
    video_frames: int
    height: int
    width: int
    dataset_type: str | None = None
    normalizer_key: str | None = None
    normalizer_min_key: str | None = None
    normalizer_delta_key: str | None = None
    normalizer_clip: bool | None = None
    coordinate_frame: str = "rfu"
    prefix_source: str = "measured_state"
    state_conditioning_required: bool | None = None
    first_frame_memory_required: bool | None = None
    stats_sha256: str | None = None
    allow_config_stats_without_checkpoint_state: bool = False
    allow_checkpoint_proprio_normalizer: bool = False
    expected_action_keys: tuple[tuple[str, int], ...] | None = None
    expected_state_keys: tuple[tuple[str, int], ...] | None = None
    prediction_start_offset: int = 0
    relative_action_chunk_anchor_mode: str | None = None
    expected_action_key_mapping: tuple[tuple[str, str], ...] | None = None
    expected_future_interval: int | None = None

    @property
    def observation_interval(self) -> int:
        if self.action_steps % self.video_frames:
            raise ValueError(
                f"profile {self.name!r} action/video ratio is not integral"
            )
        return self.action_steps // self.video_frames

    def validate_config(self, cfg: Any) -> None:
        """Reject a checkpoint config that does not implement this profile."""

        model = cfg.model
        actual = (
            int(model.action_steps_per_chunk),
            int(model.video_frames_per_chunk),
            int(model.height),
            int(model.width),
        )
        expected = (
            self.action_steps,
            self.video_frames,
            self.height,
            self.width,
        )
        if actual != expected:
            raise ValueError(
                f"VAP_ROBODOJO_PROFILE={self.name!r} requires "
                "(action_steps,video_frames,height,width)="
                f"{expected}, checkpoint config has {actual}"
            )
        if int(model.num_interleave_chunks) != 2:
            raise ValueError(
                f"profile {self.name!r} requires two IVAVA interleave chunks"
            )
        if self.dataset_type is not None and cfg.data.dataset_type != self.dataset_type:
            raise ValueError(
                f"profile {self.name!r} requires dataset_type={self.dataset_type!r}, "
                f"checkpoint config has {cfg.data.dataset_type!r}"
            )
        if self.state_conditioning_required is not None:
            actual_state_film = bool(getattr(model, "state_film_enabled", False))
            if actual_state_film != self.state_conditioning_required:
                raise ValueError(
                    f"profile {self.name!r} requires state_film_enabled="
                    f"{self.state_conditioning_required}, checkpoint has "
                    f"{actual_state_film}"
                )
        if self.first_frame_memory_required is not None:
            actual_memory = bool(getattr(model, "first_frame_memory_enabled", False))
            if actual_memory != self.first_frame_memory_required:
                raise ValueError(
                    f"profile {self.name!r} requires "
                    f"first_frame_memory_enabled="
                    f"{self.first_frame_memory_required}, checkpoint has "
                    f"{actual_memory}"
                )
        for label, attribute, expected_items in (
            ("dof_config", "dof_config", self.expected_action_keys),
            ("agent_pos_config", "agent_pos_config", self.expected_state_keys),
        ):
            if expected_items is None:
                continue
            actual_mapping = getattr(cfg.task, attribute)
            actual_items = tuple(
                (str(key), int(value)) for key, value in actual_mapping.items()
            )
            if actual_items != expected_items:
                raise ValueError(
                    f"profile {self.name!r} requires ordered task.{label}="
                    f"{expected_items!r}, checkpoint has {actual_items!r}"
                )

        norm_cfg = cfg.data.normalizer_config or {}
        expected_norm = {
            "min_key": self.normalizer_min_key,
            "delta_key": self.normalizer_delta_key,
            "clip": self.normalizer_clip,
        }
        for key, expected_value in expected_norm.items():
            if expected_value is None:
                continue
            actual_value = norm_cfg.get(key, True if key == "clip" else None)
            if actual_value != expected_value:
                raise ValueError(
                    f"profile {self.name!r} requires normalizer {key}="
                    f"{expected_value!r}, checkpoint config has {actual_value!r}"
                )
        stats_file = norm_cfg.get("customized_action_statistic_dof")
        if not stats_file:
            raise ValueError(
                f"profile {self.name!r} requires checkpoint-recorded action stats"
            )

        if self.name in EE_V3_PROFILES:
            effective = getattr(cfg.data, "dataset_config_effective", None)
            if isinstance(effective, dict):
                self._validate_effective_action_processor(effective)

    def _validate_effective_action_processor(self, effective: dict[str, Any]) -> None:
        try:
            params = effective["dataset"]["processors"]["action"]["params"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "EE-v3 checkpoint has no effective action processor"
            ) from exc
        if not isinstance(params, dict):
            raise ValueError(
                "EE-v3 effective action processor params must be a mapping"
            )
        required = {
            "prediction_start_offset": self.prediction_start_offset,
            "relative_action_chunk_size": self.action_steps,
            "rotation_source_repr": "euler",
            "rotation_repr": "rotation_6d",
        }
        for key, expected_value in required.items():
            if params.get(key) != expected_value:
                raise ValueError(
                    f"EE-v3 action processor {key}={params.get(key)!r}; "
                    f"expected {expected_value!r}"
                )
        if self.expected_future_interval is not None:
            window = effective.get("window")
            if not isinstance(window, dict):
                dataset = effective.get("dataset")
                if isinstance(dataset, dict):
                    window = dataset.get("window")
            if not isinstance(window, dict):
                raise ValueError("EE-v3 checkpoint has no effective window contract")
            actual_interval = window.get("future_interval")
            if actual_interval != self.expected_future_interval:
                raise ValueError(
                    "EE-v3 window future_interval="
                    f"{actual_interval!r}; expected "
                    f"{self.expected_future_interval!r}"
                )
            expected_horizon = self.action_steps * 2
            actual_horizon = window.get("action_horizon")
            if actual_horizon != expected_horizon:
                raise ValueError(
                    f"EE-v3 window action_horizon={actual_horizon!r}; "
                    f"expected {expected_horizon!r}"
                )
        for key, expected_items in (
            ("obs_action_keys", self.expected_state_keys),
            ("predict_action_keys", self.expected_action_keys),
        ):
            actual_items = tuple(
                (str(name), int(width)) for name, width in params.get(key, {}).items()
            )
            if actual_items != expected_items:
                raise ValueError(
                    f"EE-v3 effective {key}={actual_items!r}; "
                    f"expected {expected_items!r}"
                )
        if self.relative_action_chunk_anchor_mode is not None:
            actual_anchor_mode = params.get("relative_action_chunk_anchor_mode")
            if actual_anchor_mode != self.relative_action_chunk_anchor_mode:
                raise ValueError(
                    "EE-v3 action processor relative_action_chunk_anchor_mode="
                    f"{actual_anchor_mode!r}; expected "
                    f"{self.relative_action_chunk_anchor_mode!r}"
                )
        if self.expected_action_key_mapping is not None:
            actual_mapping = params.get("action_key_mapping")
            if not isinstance(actual_mapping, dict):
                raise ValueError("EE-v3 action processor has no action_key_mapping")
            for logical_key, expected_source in self.expected_action_key_mapping:
                actual_source = actual_mapping.get(logical_key)
                if actual_source != expected_source:
                    raise ValueError(
                        f"EE-v3 action_key_mapping[{logical_key!r}]="
                        f"{actual_source!r}; expected {expected_source!r}"
                    )

    def validate_raw_config(self, raw: Any) -> None:
        """Validate fields intentionally retained only in the saved YAML."""

        if self.name not in EE_V3_PROFILES:
            return
        if not isinstance(raw, dict):
            raise TypeError("checkpoint config.yml must decode to a mapping")
        data = raw.get("data")
        if not isinstance(data, dict):
            raise ValueError("EE-v3 checkpoint config is missing data")
        effective = data.get("dataset_config_effective")
        if not isinstance(effective, dict):
            raise ValueError(
                "EE-v3 checkpoint is missing data.dataset_config_effective"
            )
        self._validate_effective_action_processor(effective)
        if self.first_frame_memory_required:
            runtime = effective.get("_wall_x_runtime")
            memory = (
                runtime.get("first_frame_memory") if isinstance(runtime, dict) else None
            )
            model = raw.get("model")
            if not isinstance(model, dict):
                raise ValueError("Memory checkpoint config is missing model")
            history_offsets = model.get("history_offsets", [])
            if not isinstance(history_offsets, (list, tuple)):
                raise ValueError("model.history_offsets must be a list or tuple")
            expected = {
                "anchor": "resolved_episode_local_index_0",
                "enabled": True,
            }
            if history_offsets:
                expected["history_offsets"] = list(history_offsets)
            if memory != expected:
                raise ValueError(
                    "first-frame Memory profile requires effective dataset "
                    f"contract {expected!r}, checkpoint has {memory!r}"
                )

    def validate_stats_file(self, path: str | Path) -> None:
        stats_path = Path(path).expanduser().resolve()
        if not stats_path.is_file():
            raise FileNotFoundError(f"normalizer statistics not found: {stats_path}")
        if self.stats_sha256 is None:
            return
        observed = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        if observed != self.stats_sha256:
            raise ValueError(
                f"profile {self.name!r} requires stats sha256={self.stats_sha256}, "
                f"got {observed} from {stats_path}"
            )

    def validate_runtime(self, runtime: Any) -> None:
        """Validate the exact normalizer reconstructed from the checkpoint."""

        actual = (
            int(runtime.action_steps_per_chunk),
            int(runtime.video_frames_per_chunk),
        )
        expected = (self.action_steps, self.video_frames)
        if actual != expected:
            raise ValueError(
                f"profile {self.name!r} runtime horizon {actual} != {expected}"
            )
        checks = {
            "key": (runtime.normalizer_key, self.normalizer_key),
            "min_key": (runtime.normalizer_min_key, self.normalizer_min_key),
            "delta_key": (
                runtime.normalizer_delta_key,
                self.normalizer_delta_key,
            ),
            "clip": (runtime.normalizer_clip, self.normalizer_clip),
        }
        for label, (actual_value, expected_value) in checks.items():
            if expected_value is not None and actual_value != expected_value:
                raise ValueError(
                    f"profile {self.name!r} runtime normalizer {label}="
                    f"{actual_value!r} != {expected_value!r}"
                )
        if self.first_frame_memory_required is not None:
            actual_memory = bool(getattr(runtime, "first_frame_memory_enabled", False))
            if actual_memory != self.first_frame_memory_required:
                raise ValueError(
                    f"profile {self.name!r} runtime first-frame Memory="
                    f"{actual_memory} != {self.first_frame_memory_required}"
                )


_PROFILES = {
    LEGACY_PROFILE: RoboDojoEvaluationProfile(
        name=LEGACY_PROFILE,
        protocol_version=2,
        action_steps=40,
        video_frames=20,
        height=192,
        width=224,
    ),
    A32V16_320X256_PROFILE: RoboDojoEvaluationProfile(
        name=A32V16_320X256_PROFILE,
        protocol_version=3,
        action_steps=32,
        video_frames=16,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robotwin30_arx5",
        normalizer_min_key="q001",
        normalizer_delta_key="delta_001",
        normalizer_clip=True,
    ),
    EE_V3_LFU_STATE_PROFILE: RoboDojoEvaluationProfile(
        name=EE_V3_LFU_STATE_PROFILE,
        protocol_version=4,
        action_steps=32,
        video_frames=16,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robodojo_ee_v3",
        normalizer_min_key="q001",
        normalizer_delta_key="delta_001",
        normalizer_clip=True,
        coordinate_frame="lfu",
        prefix_source="issued_master",
        state_conditioning_required=True,
        stats_sha256="d7956d39c0f62d824506ac0c86cf5799f5dbca4409d720ee56ead422fbfc2058",
        allow_config_stats_without_checkpoint_state=True,
        expected_action_keys=EE_V3_ACTION_KEYS,
        expected_state_keys=EE_V3_STATE_KEYS,
    ),
    EE_V3_LFU_STATE_MINMAX_PROFILE: RoboDojoEvaluationProfile(
        name=EE_V3_LFU_STATE_MINMAX_PROFILE,
        protocol_version=5,
        action_steps=32,
        video_frames=16,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robodojo_ee_v3",
        normalizer_min_key="min",
        normalizer_delta_key="delta",
        normalizer_clip=True,
        coordinate_frame="lfu",
        prefix_source="issued_master",
        state_conditioning_required=True,
        stats_sha256="d7956d39c0f62d824506ac0c86cf5799f5dbca4409d720ee56ead422fbfc2058",
        allow_config_stats_without_checkpoint_state=True,
        expected_action_keys=EE_V3_ACTION_KEYS,
        expected_state_keys=EE_V3_STATE_KEYS,
    ),
    EE_V3_LFU_STATE_GLOBALNORM_PROFILE: RoboDojoEvaluationProfile(
        name=EE_V3_LFU_STATE_GLOBALNORM_PROFILE,
        protocol_version=6,
        action_steps=32,
        video_frames=16,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robodojo_ee_v3",
        normalizer_min_key="q001",
        normalizer_delta_key="delta_001",
        normalizer_clip=True,
        coordinate_frame="lfu",
        prefix_source="issued_master",
        state_conditioning_required=True,
        stats_sha256="bc30897670ed5210b66dcc52aaf39bc9ea7726be56f1d7ff9c3c08d48fc73f75",
        allow_config_stats_without_checkpoint_state=True,
        allow_checkpoint_proprio_normalizer=True,
        expected_action_keys=EE_V3_ACTION_KEYS,
        expected_state_keys=EE_V3_STATE_KEYS,
    ),
    EE_V3_LFU_NOSTATE_PROFILE: RoboDojoEvaluationProfile(
        name=EE_V3_LFU_NOSTATE_PROFILE,
        protocol_version=4,
        action_steps=32,
        video_frames=16,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robodojo_ee_v3",
        normalizer_min_key="q001",
        normalizer_delta_key="delta_001",
        normalizer_clip=True,
        coordinate_frame="lfu",
        prefix_source="issued_master",
        state_conditioning_required=False,
        stats_sha256="d7956d39c0f62d824506ac0c86cf5799f5dbca4409d720ee56ead422fbfc2058",
        allow_config_stats_without_checkpoint_state=True,
        expected_action_keys=EE_V3_ACTION_KEYS,
        expected_state_keys=EE_V3_STATE_KEYS,
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE: RoboDojoEvaluationProfile(
        name=EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE,
        protocol_version=6,
        action_steps=32,
        video_frames=16,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robodojo_ee_v3",
        normalizer_min_key="q001",
        normalizer_delta_key="delta_001",
        normalizer_clip=True,
        coordinate_frame="lfu",
        prefix_source="issued_master",
        state_conditioning_required=False,
        first_frame_memory_required=False,
        stats_sha256="bc30897670ed5210b66dcc52aaf39bc9ea7726be56f1d7ff9c3c08d48fc73f75",
        allow_config_stats_without_checkpoint_state=True,
        expected_action_keys=EE_V3_ACTION_KEYS,
        expected_state_keys=EE_V3_STATE_KEYS,
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE: RoboDojoEvaluationProfile(
        name=EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE,
        protocol_version=7,
        action_steps=32,
        video_frames=16,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robodojo_ee_v3",
        normalizer_min_key="q001",
        normalizer_delta_key="delta_001",
        normalizer_clip=True,
        coordinate_frame="lfu",
        prefix_source="measured_state",
        state_conditioning_required=False,
        first_frame_memory_required=False,
        stats_sha256="bc30897670ed5210b66dcc52aaf39bc9ea7726be56f1d7ff9c3c08d48fc73f75",
        allow_config_stats_without_checkpoint_state=True,
        expected_action_keys=EE_V3_ACTION_KEYS,
        expected_state_keys=EE_V3_STATE_KEYS,
        prediction_start_offset=1,
        relative_action_chunk_anchor_mode="observation_aligned",
        expected_action_key_mapping=EE_V3_FOLLOWER_ACTION_KEY_MAPPING,
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE: RoboDojoEvaluationProfile(
        name=EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE,
        protocol_version=6,
        action_steps=32,
        video_frames=16,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robodojo_ee_v3",
        normalizer_min_key="q001",
        normalizer_delta_key="delta_001",
        normalizer_clip=True,
        coordinate_frame="lfu",
        prefix_source="issued_master",
        state_conditioning_required=False,
        first_frame_memory_required=True,
        stats_sha256="bc30897670ed5210b66dcc52aaf39bc9ea7726be56f1d7ff9c3c08d48fc73f75",
        allow_config_stats_without_checkpoint_state=True,
        expected_action_keys=EE_V3_ACTION_KEYS,
        expected_state_keys=EE_V3_STATE_KEYS,
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE: RoboDojoEvaluationProfile(
        name=EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE,
        protocol_version=9,
        action_steps=32,
        video_frames=8,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robodojo_ee_v3",
        normalizer_min_key="q001",
        normalizer_delta_key="delta_001",
        normalizer_clip=True,
        coordinate_frame="lfu",
        prefix_source="issued_master",
        state_conditioning_required=False,
        first_frame_memory_required=True,
        stats_sha256="bc30897670ed5210b66dcc52aaf39bc9ea7726be56f1d7ff9c3c08d48fc73f75",
        allow_config_stats_without_checkpoint_state=True,
        expected_action_keys=EE_V3_ACTION_KEYS,
        expected_state_keys=EE_V3_STATE_KEYS,
        expected_future_interval=4,
    ),
    EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE: RoboDojoEvaluationProfile(
        name=EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE,
        protocol_version=8,
        action_steps=64,
        video_frames=16,
        height=256,
        width=320,
        dataset_type="x2robot_v2",
        normalizer_key="robodojo_ee_v3",
        normalizer_min_key="q001",
        normalizer_delta_key="delta_001",
        normalizer_clip=True,
        coordinate_frame="lfu",
        prefix_source="issued_master",
        state_conditioning_required=False,
        first_frame_memory_required=False,
        stats_sha256="bc30897670ed5210b66dcc52aaf39bc9ea7726be56f1d7ff9c3c08d48fc73f75",
        allow_config_stats_without_checkpoint_state=True,
        expected_action_keys=EE_V3_ACTION_KEYS,
        expected_state_keys=EE_V3_STATE_KEYS,
        expected_future_interval=4,
    ),
}


def resolve_evaluation_profile(value: Any) -> RoboDojoEvaluationProfile:
    raw = str(value or LEGACY_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "legacy": LEGACY_PROFILE,
        "a40v20": LEGACY_PROFILE,
        "a40v20_192x224": LEGACY_PROFILE,
        "a32v16": A32V16_320X256_PROFILE,
        "ivava16a32_320x256": A32V16_320X256_PROFILE,
        "eev3": EE_V3_LFU_STATE_PROFILE,
        "ee_v3": EE_V3_LFU_STATE_PROFILE,
        "a32v16_eev3": EE_V3_LFU_STATE_PROFILE,
        "eev3_minmax": EE_V3_LFU_STATE_MINMAX_PROFILE,
        "ee_v3_minmax": EE_V3_LFU_STATE_MINMAX_PROFILE,
        "a32v16_eev3_minmax": EE_V3_LFU_STATE_MINMAX_PROFILE,
        "eev3_state_globalnorm": EE_V3_LFU_STATE_GLOBALNORM_PROFILE,
        "ee_v3_state_globalnorm": EE_V3_LFU_STATE_GLOBALNORM_PROFILE,
        "a32v16_eev3_state_globalnorm": EE_V3_LFU_STATE_GLOBALNORM_PROFILE,
        "eev3_nostate": EE_V3_LFU_NOSTATE_PROFILE,
        "ee_v3_nostate": EE_V3_LFU_NOSTATE_PROFILE,
        "a32v16_eev3_nostate": EE_V3_LFU_NOSTATE_PROFILE,
        "eev3_nostate_globalnorm": EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE,
        "ee_v3_nostate_globalnorm": EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE,
        "a32v16_eev3_nostate_globalnorm": EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE,
        "eev3_nostate_globalnorm_follower": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE
        ),
        "ee_v3_nostate_globalnorm_follower": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE
        ),
        "a32v16_eev3_nostate_globalnorm_follower": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE
        ),
        "eev3_nostate_globalnorm_memory": (EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE),
        "ee_v3_nostate_globalnorm_memory": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE
        ),
        "a32v16_eev3_nostate_globalnorm_memory": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE
        ),
        "a64v16_eev3_nostate_globalnorm": (EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE),
        "a32v8_eev3_nostate_globalnorm_memory": (
            EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE
        ),
        "eev3_nostate_globalnorm_a64v16": (EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE),
    }
    name = aliases.get(raw, raw)
    if name not in _PROFILES:
        raise ValueError(
            f"unsupported VAP_ROBODOJO_PROFILE={value!r}; "
            f"available={sorted(_PROFILES)}"
        )
    return _PROFILES[name]


__all__ = [
    "A32V16_320X256_PROFILE",
    "EE_V3_LFU_STATE_PROFILE",
    "EE_V3_LFU_STATE_MINMAX_PROFILE",
    "EE_V3_LFU_STATE_GLOBALNORM_PROFILE",
    "EE_V3_LFU_NOSTATE_PROFILE",
    "EE_V3_LFU_NOSTATE_GLOBALNORM_PROFILE",
    "EE_V3_LFU_NOSTATE_GLOBALNORM_FOLLOWER_PROFILE",
    "EE_V3_LFU_NOSTATE_GLOBALNORM_MEMORY_PROFILE",
    "EE_V3_LFU_NOSTATE_GLOBALNORM_A32V8_MEMORY_PROFILE",
    "EE_V3_LFU_NOSTATE_GLOBALNORM_A64V16_PROFILE",
    "LEGACY_PROFILE",
    "RoboDojoEvaluationProfile",
    "resolve_evaluation_profile",
]
