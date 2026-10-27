from typing import Any

from omegaconf import DictConfig, OmegaConf


def _select(cfg: DictConfig | dict[str, Any], path: str, default: Any = None) -> Any:
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, path, default=default)
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _positive_int(value: Any, path: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"`{path}` must be positive, got {parsed}.")
    return parsed


def _dataset_sampling_value(
    cfg: DictConfig | dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if bool(_select(cfg, "model.joint96_three_view", False)):
        return _select(cfg, f"data.sampling.{key}", default)
    direct = _select(cfg, f"data.train.{key}", None)
    if direct is not None:
        return direct
    dataset_config = _select(cfg, "data.train.dataset_config", None)
    if not dataset_config:
        return default
    loaded = OmegaConf.load(str(dataset_config))
    return OmegaConf.select(loaded, f"sampling.{key}", default=default)


def validate_training_config(cfg: DictConfig | dict[str, Any]) -> None:
    """Fail fast on the frame-window Phoenix-MWA training contract."""
    from wam.training.multiview_recipe import MULTIVIEW_READER, validate_multiview_config

    validate_multiview_config(cfg)
    if bool(_select(cfg, "joint96_posttrain.enabled", False)) and not bool(_select(cfg, "model.joint96_three_view", False)):
        raise ValueError("Joint96 posttraining cannot disable model.joint96_three_view")
    compile_mode = str(_select(cfg, "model.mot_compile_mode", "off"))
    if compile_mode not in {"off", "default", "reduce-overhead"}:
        raise ValueError("model.mot_compile_mode must be off, default or reduce-overhead.")
    compile_gc = bool(_select(cfg, "model.mot_compile_gradient_checkpointing", False))
    if compile_gc and compile_mode == "off":
        raise ValueError("model.mot_compile_gradient_checkpointing requires compilation.")
    if compile_mode != "off":
        for path in ("model.mot_checkpoint_mixed_attn", "model.video_dit_config.use_gradient_checkpointing", "model.action_dit_config.use_gradient_checkpointing"):
            if bool(_select(cfg, path, False)):
                raise ValueError(f"MoT compilation requires {path}=false; use mot_compile_gradient_checkpointing for recomputation.")
        if bool(_select(cfg, "model.understanding.enabled", False)) and int(_select(cfg, "model.mot_compile_action_context_pad_to", 0)) <= 0:
            raise ValueError("Compiled VLM conditioning requires positive model.mot_compile_action_context_pad_to.")
    cache_mode = str(_select(cfg, "video_latent_cache", "off"))
    if cache_mode != "off":
        for name in ("image_color_jitter", "image_camera_jitter", "image_sensor_noise", "image_domain_jitter"):
            if bool(_select(cfg, f"data.train.{name}.enabled", False)):
                raise ValueError(f"Offline encoder artifacts require deterministic images; data.train.{name} is enabled.")

    if bool(_select(cfg, "model.memory.enabled", False)):
        raise ValueError(
            "Frame-window training requires model.memory.enabled=false."
        )

    if bool(_select(cfg, "model.distillation.enabled", False)):
        if _select(cfg, "data.train._target_", "") != "wam.datasets.distillation_dataset.DistillationDataset":
            raise ValueError("4D distillation requires DistillationDataset.")
        batch_size = _positive_int(_select(cfg, "batch_size"), "batch_size")
        count = _positive_int(_select(cfg, "data.train.teacher_samples_per_batch", 1), "teacher_samples_per_batch")
        if count > batch_size:
            raise ValueError("teacher_samples_per_batch exceeds batch_size.")
        if int(_select(cfg, "model.distillation.teacher.feature_dim", 1430)) != 1430:
            raise ValueError("The 4D cache contract requires 1430 teacher features.")
        cache = _select(cfg, "data.train.teacher_cache", None)
        if not cache:
            raise ValueError("Set WAM_4D_CACHE to a completed teacher cache directory.")
        from wam.datasets.distillation_cache import TeacherCache
        TeacherCache(str(cache))

    num_frames = _positive_int(
        _dataset_sampling_value(cfg, "num_frames"),
        "data.train.num_frames",
    )
    action_horizon = int(
        _dataset_sampling_value(cfg, "action_size", num_frames - 1) or num_frames - 1
    )
    if action_horizon != num_frames - 1:
        raise ValueError(
            "Frame training requires one action per RGB-window transition: "
            f"got num_frames={num_frames}, action_horizon={action_horizon}."
        )

    rtc_enabled_path = "model.action_dit_config.training_time_rtc.enabled"
    rtc_enabled = _select(cfg, rtc_enabled_path, False)
    if not isinstance(rtc_enabled, bool):
        raise ValueError(
            f"`{rtc_enabled_path}` must be a boolean, got {rtc_enabled!r}."
        )
    if rtc_enabled:
        max_delay_path = (
            "model.action_dit_config.training_time_rtc.max_delay_steps"
        )
        max_delay_steps = _select(cfg, max_delay_path, 16)
        if isinstance(max_delay_steps, bool) or not isinstance(
            max_delay_steps, int
        ):
            raise ValueError(
                f"`{max_delay_path}` must be an integer, "
                f"got {max_delay_steps!r}."
            )
        if not 0 <= max_delay_steps < action_horizon:
            raise ValueError(
                f"`{max_delay_path}` must satisfy "
                f"0 <= max_delay_steps < action_horizon ({action_horizon}), "
                f"got {max_delay_steps}."
            )

    anchor_frames = int(
        _select(cfg, "model.memory.video.num_anchor_frames", 0) or 0
    )
    recent_frames = int(
        _select(cfg, "model.memory.video.num_recent_frames", 0) or 0
    )
    data_anchor_frames = int(
        _dataset_sampling_value(cfg, "memory_video_anchor_size", 0) or 0
    )
    recent_offset = int(
        _dataset_sampling_value(cfg, "memory_recent_frame_offset", 0) or 0
    )
    joint96 = bool(_select(cfg, "model.joint96_three_view", False))
    if joint96:
        if bool(_select(cfg, "model.memory.video.enabled", True)) or (anchor_frames, recent_frames, data_anchor_frames, recent_offset) != (0, 0, 0, 0):
            raise ValueError("Local joint96 requires current-only observations, without memory video histories")
        posttrain = bool(_select(cfg, "joint96_posttrain.enabled", False))
        expected_reader = ("wam.datasets.joint96_posttrain.Joint96PosttrainDataset" if posttrain
                           else "wam.datasets.local_robot.LocalRobot80Dataset")
        frame_ablation = _select(cfg, "data.frame_ablation")
        if _select(cfg, "robot36.enabled", False):
            if posttrain or frame_ablation is not None:
                raise ValueError("Robot36 is an independent raw-field pretraining recipe")
            if _select(cfg, "robot36.contract") != "phoenix.local_robot36.raw_panda_b.v1":
                raise ValueError("Robot36 contract must be explicitly versioned")
            from wam.datasets.local_robot_main_view_aug import validate_config
            for split in ("train", "val"):
                validate_config(_select(cfg, f"data.{split}.main_view_augmentation"))
            if _select(cfg, "data.val.main_view_augmentation.enabled", False):
                raise ValueError("Robot36 validation cannot enable main-view augmentation")
            expected_reader = "wam.datasets.local_robot36.Robot36Dataset"
        if _select(cfg, "robot37.enabled", False):
            if posttrain or frame_ablation is not None or _select(cfg, "robot36.enabled", False):
                raise ValueError("Robot37 requires its independent pretraining recipe")
            if _select(cfg, "robot37.contract") != "phoenix.local_robot37.rh20t_v6.v1":
                raise ValueError("Robot37 contract must be explicitly versioned")
            from wam.datasets.local_robot_main_view_aug import validate_config
            for split in ("train", "val"):
                validate_config(_select(cfg, f"data.{split}.main_view_augmentation"))
            if _select(cfg, "data.val.main_view_augmentation.enabled", False):
                raise ValueError("Robot37 validation cannot enable augmentation")
            expected_reader = "wam.datasets.local_robot37.Robot37Dataset"
        if _select(cfg, "data.train._target_") == "wam.datasets.local_robot_main_view_aug.LocalRobot80MainAugDataset":
            if posttrain or frame_ablation is not None:
                raise ValueError("Main augmentation is an independent LocalRobot80 pretraining recipe")
            from wam.datasets.local_robot_main_view_aug import validate_config
            for split in ("train", "val"):
                options = _select(cfg, f"data.{split}.main_view_augmentation")
                validate_config(options)
            if _select(cfg, "data.val.main_view_augmentation.enabled", False):
                raise ValueError("Main-camera augmentation must be disabled in validation")
            expected_reader = "wam.datasets.local_robot_main_view_aug.LocalRobot80MainAugDataset"
        if frame_ablation is not None:
            frame_readers = {
                "remove_world_A_keep_B_v1": "wam.datasets.local_robot_no_a.LocalRobot80NoADataset",
                "remove_world_A_and_local_B_v1": "wam.datasets.local_robot_no_ab.LocalRobot80NoABDataset",
            }
            if (posttrain or not _select(cfg, "local_robot_joint96.enabled", False)
                    or frame_ablation not in frame_readers):
                raise ValueError("Unsupported Joint96 frame ablation")
            expected_reader = frame_readers[frame_ablation]
        cached_visuals = bool(_select(cfg, "joint96_posttrain.cached_visuals", False))
        robotwin_prompt_posttrain = bool(_select(cfg, "joint96_posttrain.robotwin_softprompt.enabled", False))
        robodojo_real_posttrain = bool(_select(cfg, "joint96_posttrain.robodojo_real_softprompt.enabled", False))
        if robodojo_real_posttrain:
            if not posttrain or cached_visuals:
                raise ValueError("RoboDojo real Joint96 requires explicit online weights-only posttraining")
            from wam.training.robodojo_real_softprompt import validate_robodojo_real_prompt
            validate_robodojo_real_prompt(cfg)
            expected_reader = "wam.datasets.robodojo_real_joint96.RoboDojoRealJoint96Dataset"
        if robotwin_prompt_posttrain:
            if not posttrain or cached_visuals or not _select(cfg, "joint96_posttrain.online_robotwin", False):
                raise ValueError("Robot37 RoboTwin prompt recipe must use explicit online posttraining")
            expected_reader = "wam.datasets.robotwin_joint96_robot37.RobotwinJoint96OnlineDataset"
            for split in ("train", "val"):
                if _select(cfg, f"data.{split}.domain") != "robotwin" or _select(cfg, f"data.{split}.visual_cache") is not None:
                    raise ValueError("Online RoboTwin requires matching domains without visual caches")
            if (_select(cfg, "joint96_posttrain.evaluation_membership") != "train_pool"
                    or float(_select(cfg, "data.train.val_set_proportion", 0)) != 0):
                raise ValueError("Official RoboTwin uses all clean episodes; diagnostics are not held out")
            from wam.model.modules.conditioning.frame_attention import JOINT96_ACTION_VIDEO_ATTENTION
            if _select(cfg, "joint96_posttrain.action_video_attention") != JOINT96_ACTION_VIDEO_ATTENTION:
                raise ValueError("RoboTwin must match current/noisy-future Joint attention")
        robocasa = bool(_select(cfg, "joint96_posttrain.robocasa.enabled", False))
        if robocasa:
            from wam.training.robocasa_posttrain import validate_robocasa_config
            validate_robocasa_config(cfg)
        if cached_visuals:
            if not posttrain:
                raise ValueError("External LIBERO cache requires Joint96 posttraining")
            expected_reader = ("wam.datasets.robocasa_posttrain.RoboCasaPosttrainDataset" if robocasa
                               else "wam.datasets.libero_joint96_cache.LiberoJoint96CachedDataset")
            for split in ("train", "val"):
                if _select(cfg, f"data.{split}.domain") != ("robocasa" if robocasa else "libero"):
                    raise ValueError("External Joint96 visual cache domain mismatch")
                for name in ("image_color_jitter", "image_camera_jitter", "image_sensor_noise", "image_domain_jitter"):
                    if bool(_select(cfg, f"data.{split}.{name}.enabled", False)):
                        raise ValueError(f"Fixed Joint96 visual cache requires data.{split}.{name}.enabled=false")
        for split in ("train", "val"):
            split_reader = expected_reader
            if _select(cfg, "local_robot_multiview.enabled", False) and (
                split == "train"
                or _select(cfg, "local_robot_multiview.rgb_selection_root") is not None
            ):
                split_reader = MULTIVIEW_READER
            if _select(cfg, f"data.{split}._target_") != split_reader:
                raise ValueError("Local joint96 requires explicit local Robot80 readers for both splits")
        if posttrain:
            if _select(cfg, "data.train.domain") == "libero":
                from wam.model.modules.conditioning.frame_attention import JOINT96_ACTION_VIDEO_ATTENTION
                if _select(cfg, "joint96_posttrain.action_video_attention") != JOINT96_ACTION_VIDEO_ATTENTION:
                    raise ValueError(
                        "LIBERO Joint96 posttraining requires joint96_posttrain.action_video_attention="
                        + JOINT96_ACTION_VIDEO_ATTENTION
                    )
            if _select(cfg, "local_robot_joint96.enabled", False) or _select(cfg, "resume") is not None:
                raise ValueError("Joint96 posttraining is a new weights-only run, not local pretraining/resume")
            if not _select(cfg, "init_checkpoint"):
                raise ValueError("Joint96 posttraining requires init_checkpoint")
            if _select(cfg, "video_latent_cache", "off") != "off":
                raise ValueError("Joint96 posttraining uses its explicit reader, not the native CacheManager")
            for path, expected in (("model.video_dit_config.in_dim", 96), ("model.video_dit_config.out_dim", 96),
                                   ("model.action_dit_config.action_dim", 80), ("model.proprio_dim", 80)):
                if int(_select(cfg, path, 0)) != expected:
                    raise ValueError(f"Joint96 posttraining requires {path}={expected}")
            if int(_select(cfg, "eval_every", 0)) > 0:
                if not (robotwin_prompt_posttrain or robodojo_real_posttrain) and float(_select(cfg, "data.train.val_set_proportion", 0)) <= 0:
                    raise ValueError("Joint96 held-out eval requires a positive episode holdout fraction")
    elif (anchor_frames, recent_frames, data_anchor_frames) != (1, 1, 1):
        raise ValueError(
            "Frame conditioning requires one model anchor, one model recent, "
            f"and one dataset anchor; got {(anchor_frames, recent_frames, data_anchor_frames)}."
        )
    if not joint96 and recent_offset != action_horizon:
        raise ValueError(
            "The recent frame must be the previous decision observation: "
            f"offset={recent_offset}, action_horizon={action_horizon}."
        )

    sampling_ratio = _positive_int(
        _dataset_sampling_value(cfg, "action_video_freq_ratio", 4),
        "data.train.action_video_freq_ratio",
    )
    if action_horizon % sampling_ratio != 0:
        raise ValueError(
            "Action horizon must be divisible by action_video_freq_ratio: "
            f"horizon={action_horizon}, ratio={sampling_ratio}."
        )
    sampled_video_frames = action_horizon // sampling_ratio + 1
    if sampled_video_frames % 4 != 1:
        raise ValueError(
            "The sampled RGB window must satisfy T % 4 == 1 for the Wan VAE: "
            f"horizon={action_horizon}, ratio={sampling_ratio}, "
            f"sampled_frames={sampled_video_frames}."
        )

    attention_backend = str(_select(cfg, "model.mot_attention_backend", "flex")).lower()
    if _select(cfg, "joint96_posttrain.robocasa.enabled", False):
        if (attention_backend != "sdpa" or _select(cfg, "model.mot_flex_masked_batches", False)
                or str(_select(cfg, "model.mot_compile_mode", "off")).lower() != "off"):
            raise ValueError("RoboCasa reproduction requires SDPA, masked flex off, compile off")
    elif attention_backend != "flex":
        raise ValueError("Frame Phoenix-MWA currently requires model.mot_attention_backend=flex.")
    for field in ("num_layers", "num_heads", "attn_head_dim"):
        video_value = int(_select(cfg, f"model.video_dit_config.{field}"))
        action_value = int(_select(cfg, f"model.action_dit_config.{field}"))
        if video_value != action_value:
            raise ValueError(
                f"MoT video/action {field} must match, got "
                f"video={video_value}, action={action_value}."
            )

    if bool(_select(cfg, "model.future_delta.enabled", False)):
        future_delta_num_tokens = _select(
            cfg, "model.future_delta.num_tokens", None
        )
        if future_delta_num_tokens is not None:
            _positive_int(
                future_delta_num_tokens, "model.future_delta.num_tokens"
            )
        else:
            layout = str(
                _select(cfg, "data.train.concat_multi_camera", "single") or "single"
            ).lower()
            num_cameras = int(
                _select(cfg, "data.train.processor.num_output_cameras", 1) or 1
            )
            expected_views = num_cameras if layout == "latent_horizontal" else 1
            configured_views = _positive_int(
                _select(cfg, "model.future_delta.num_views", 1),
                "model.future_delta.num_views",
            )
            if configured_views != expected_views:
                raise ValueError(
                    "Future Delta queries must match the post-layout video raster: "
                    f"layout={layout!r} expects num_views={expected_views}, "
                    f"got {configured_views}."
                )
        future_delta_weight = float(
            _select(cfg, "model.future_delta.loss_weight", 0.5)
        )
        if future_delta_weight < 0.0:
            raise ValueError(
                "model.future_delta.loss_weight must be non-negative."
            )

    if bool(_select(cfg, "model.understanding.enabled", False)):
        if not str(_select(cfg, "model.understanding.vlm_model_path", "")):
            raise ValueError(
                "model.understanding.vlm_model_path is required when understanding is enabled."
            )
        vlm_batch_size = int(
            _select(cfg, "model.understanding.vlm_batch_size", 0) or 0
        )
        if vlm_batch_size < 0:
            raise ValueError(
                f"model.understanding.vlm_batch_size must be >= 0, got {vlm_batch_size}."
            )
        if bool(_select(cfg, "model.understanding.train_vlm", False)):
            vlm_learning_rate = float(
                _select(cfg, "vlm_learning_rate", 5.0e-5)
            )
            if not 0.0 < vlm_learning_rate <= 5.0e-5:
                raise ValueError(
                    "Trainable VLM learning rate must be in (0, 5e-5], "
                    f"got {vlm_learning_rate}."
                )
