from __future__ import annotations

from itertools import product
from pathlib import Path

from omegaconf import OmegaConf

from worldscape_policy.cli.config_composition import load_composed_config
from worldscape_policy.cli.config_profiles import resolve_config_profiles
from worldscape_policy.model_config import GenerationConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
MODEL_GRAPH = "../model/wsp2_wan22_5b.yaml"
WAM_GRAPH = "../wam/wan22_5b.yaml"


def test_pretrain_and_posttrain_share_model_and_wam_without_inline_duplication(
    monkeypatch,
):
    monkeypatch.delenv("ACTION_HORIZON", raising=False)
    pretrain_path = CONFIGS / "pretrain" / "mixed_three_mode_wan22.yaml"
    posttrain_path = CONFIGS / "posttrain" / "common_wan22.yaml"
    pretrain_raw = OmegaConf.load(pretrain_path)
    posttrain_raw = OmegaConf.load(posttrain_path)
    pretrain_common_raw = OmegaConf.load(
        CONFIGS / "pretrain" / "common_wan22.yaml"
    )

    assert list(pretrain_raw.includes) == [
        "common_wan22.yaml",
        MODEL_GRAPH,
        "../conditioning/interactive_t5.yaml",
        "../memory/event_default.yaml",
        "../memory/visual_prefill_default.yaml",
        WAM_GRAPH,
    ]
    assert list(pretrain_common_raw.includes) == [
        "../distributed/deepspeed.yaml",
        "../embodiment/pretrain_multi_embodiment_adapter_ids.yaml",
    ]
    assert "profiles" not in pretrain_common_raw
    assert "profiles" not in posttrain_raw
    assert list(posttrain_raw.includes) == [
        "../distributed/deepspeed.yaml",
        "../embodiment/pretrain_multi_embodiment_adapter_ids.yaml",
        MODEL_GRAPH,
        "../conditioning/interactive_t5.yaml",
        "../conditioning/auto_qwen3vl.yaml",
        "../memory/event_default.yaml",
        "../memory/visual_prefill_default.yaml",
        WAM_GRAPH,
    ]
    assert "_target_" not in pretrain_raw.model
    assert "noise_kernel" not in pretrain_raw

    pretrain = load_composed_config(pretrain_path)
    posttrain = load_composed_config(posttrain_path)
    assert pretrain.model._target_ == (
        "worldscape_policy.native_builder.build_wan22_policy_from_checkpoint"
    )
    assert pretrain.model.expected_mode == "interactive"
    assert pretrain.model.model_config.model.mode == "interactive"
    assert pretrain.model.model_config.model.event_memory.enabled is False
    assert pretrain.data_loader.mode == "interactive"
    assert pretrain.objective.semantic_forcing_weight == 0.0
    assert pretrain.prompt_schedule.enabled is False
    assert OmegaConf.to_container(pretrain.noise_kernel, resolve=False) == (
        OmegaConf.to_container(posttrain.noise_kernel, resolve=False)
    )
    assert OmegaConf.to_container(pretrain.noise_kernel.config, resolve=True) == {
        "_target_": "worldscape_policy.wam.wan22.Wan22KernelConfig",
        "num_train_timesteps": 1000,
        "num_inference_steps": 50,
        "num_frame_per_block": 2,
        "num_action_per_block": 24,
        "num_state_per_block": 1,
        "action_horizon": 24,
    }


def test_module_fragments_exclusively_own_model_subtrees():
    model = OmegaConf.load(CONFIGS / "model" / "wsp2_wan22_5b.yaml")
    interactive = OmegaConf.load(
        CONFIGS / "conditioning" / "interactive_t5.yaml"
    )
    auto = OmegaConf.load(CONFIGS / "conditioning" / "auto_qwen3vl.yaml")
    event = OmegaConf.load(CONFIGS / "memory" / "event_default.yaml")
    visual = OmegaConf.load(
        CONFIGS / "memory" / "visual_prefill_default.yaml"
    )
    wam = OmegaConf.load(CONFIGS / "wam" / "wan22_5b.yaml")

    generic = model.model.model_config.model
    assert set(generic) == {"mode", "shape"}
    for forbidden in (
        "condition_router",
        "event_memory",
        "visual_memory",
        "wam",
    ):
        assert forbidden not in generic
    assert "generation_config" not in model.model
    assert "diffusion_model_pretrained_path" not in model.model
    assert "text_encoder_pretrained_path" not in model.model
    assert "image_encoder_pretrained_path" not in model.model
    assert "vae_pretrained_path" not in model.model
    assert "vlm_pretrained_path" not in model.model

    assert set(
        interactive.model.model_config.model.condition_router
    ) == {"interactive"}
    assert "text_encoder_pretrained_path" in interactive.model
    assert set(auto.model.model_config.model.condition_router) == {"auto"}
    assert "vlm_pretrained_path" in auto.model
    assert "freeze" not in auto
    auto_vlm = auto.model.model_config.model.condition_router.auto.vlm.parameters
    assert "freeze_vlm" not in auto_vlm
    assert "freeze_qformer" not in auto_vlm
    assert set(event.model.model_config.model.event_memory) == {
        "enabled",
        "history_steps",
        "global_slots",
        "local_steps",
        "boundary_steps",
        "boundary_min_gap",
        "perception_gist_tokens",
        "residual_scale",
        "dropout",
    }
    assert "vae" in visual.model.model_config.model.visual_memory
    assert "image_encoder" in visual.model.model_config.model.visual_memory
    assert "vae_pretrained_path" in visual.model
    assert "image_encoder_pretrained_path" in visual.model
    assert wam.model.model_config.model.wam.plugin == "wan22"
    assert wam.model.model_config.model.wam.variant == "ti2v-5b"
    assert wam.model.generation_config.schema_version == "1"
    assert "diffusion_model_pretrained_path" in wam.model


def test_posttrain_runtime_and_condition_width_have_single_resolved_values(
    monkeypatch,
):
    monkeypatch.delenv("DISTRIBUTED_BACKEND", raising=False)
    monkeypatch.delenv("PRETRAINED_MODEL_FORMAT", raising=False)
    monkeypatch.delenv("DATALOADER_NUM_WORKERS", raising=False)
    monkeypatch.delenv("DATALOADER_PREFETCH_FACTOR", raising=False)
    monkeypatch.delenv("DATALOADER_PERSISTENT_WORKERS", raising=False)
    monkeypatch.setenv("SEED", "123")
    monkeypatch.setenv("VLM_CONTEXT_DIM", "3072")
    config = resolve_config_profiles(
        load_composed_config(CONFIGS / "posttrain" / "agilex.yaml"),
        overrides=OmegaConf.from_dotlist(
            [
                "selectors.mode=auto",
                "selectors.visual_prompt=none",
                "selectors.dataset_name=worldscape_hdf5_text",
                "selectors.deepspeed_mode=zero2",
            ]
        ),
    )

    shape = config.model.model_config.model.shape
    auto = config.model.model_config.model.condition_router.auto
    core = config.model.model_config.model.wam.core.parameters
    assert config.distributed.backend == "deepspeed"
    assert config.distributed.seed == 123
    assert shape.condition_dim == 3072
    assert auto.projector.output_dim == shape.condition_dim
    assert core.text_dim == shape.condition_dim
    assert config.data_loader.num_workers == 2
    assert config.data_loader.shuffle is False
    assert config.data_loader.bucket_by_length is True
    assert config.data_loader.dataset_kwargs.shard_size == 10_000
    assert config.data_loader.dataset_kwargs.shard_sampling_rate == 0.1
    assert config.data_loader.dataset_kwargs.num_shards_to_sample == 2**20
    assert config.data_loader.prefetch_factor == 2
    assert config.data_loader.persistent_workers is True

def test_v004_source_rows_initialize_single_adapter_posttrain_models():
    source_rows = {"agilex": 2, "libero": 4, "robotwin2": 4}
    posttrain_configs = {
        name: load_composed_config(CONFIGS / "posttrain" / f"{name}.yaml")
        for name in source_rows
    }

    for name, config in posttrain_configs.items():
        assert dict(config.pretrained_adapter_source_rows) == source_rows
        assert dict(config.batch_adapter.embodiment_ids) == {name: 0}
        assert config.batch_adapter.diffusion_view_layout == "mosaic_2x2"
        assert config.model.pretrained_action_adapter_index == source_rows[name]
        assert (
            config.model.model_config.model.wam.core.parameters.max_num_embodiments
            == 1
        )
        exported = config.native_export.transform_bundle.embodiments[name]
        assert exported.embodiment_id == 0

    libero = load_composed_config(CONFIGS / "eval" / "libero.yaml")
    assert "adapter" not in libero.backend_config
    robotwin_manager = OmegaConf.load(
        CONFIGS / "eval" / "robotwin2_manager.yaml"
    )
    assert robotwin_manager.EVALUATION.policy_name == "wsp2_policy"
    assert robotwin_manager.EVALUATION.action_horizon == 24
    assert not (CONFIGS / "eval" / "robotwin2.yaml").exists()

    for filename in (
        "mixed_three_mode_wan22.yaml",
        "mixed_three_mode_wan22_stage2.yaml",
    ):
        config = load_composed_config(CONFIGS / "pretrain" / filename)
        assert config.batch_adapter.embodiment_ids.agilex == source_rows["agilex"]
        assert config.batch_adapter.diffusion_view_layout == "mosaic_2x2"
        assert (
            config.model.model_config.model.wam.core.parameters.max_num_embodiments
            == 8
        )


def test_robotwin_horizon48_resolves_one_48_action_block(monkeypatch):
    monkeypatch.setenv("ACTION_HORIZON", "48")
    monkeypatch.setenv("WAM_MAX_CHUNK_SIZE", "1")
    monkeypatch.setenv("WAM_NUM_FRAMES", "9")

    config = load_composed_config(CONFIGS / "posttrain" / "robotwin2.yaml")
    shape = config.model.model_config.model.shape
    core = config.model.model_config.model.wam.core.parameters
    kernel = config.noise_kernel.config

    assert shape.num_frames == 9
    assert shape.action_horizon == 48
    assert shape.actions_per_block == 48
    assert core.num_action_per_block == 48
    assert core.max_chunk_size == 1
    assert kernel.action_horizon == 48
    assert kernel.num_action_per_block == 48
    assert config.data_loader.dataset_kwargs.action_horizon == 48
    ModelConfig.from_dict(
        OmegaConf.to_container(config.model.model_config, resolve=True)
    )


def test_robotwin_horizon48_eval_geometry(monkeypatch):
    monkeypatch.setenv("ROBOTWIN_ACTION_HORIZON", "48")
    monkeypatch.delenv("ROBOTWIN_REPLAN_STEPS", raising=False)
    monkeypatch.setenv("ROBOTWIN_OBSERVATION_INTERVAL", "6")
    monkeypatch.delenv("ROBOTWIN_VLM_HISTORY_STRIDE", raising=False)

    config = OmegaConf.load(CONFIGS / "eval" / "robotwin2_manager.yaml")

    assert config.EVALUATION.action_horizon == 48
    assert config.EVALUATION.replan_steps == 48
    assert config.EVALUATION.observation_interval == 6
    assert config.EVALUATION.vlm_history_stride == 48


def test_robotwin_concat_layout_is_shared_and_old_configs_default_to_mosaic(
    monkeypatch,
):
    default_config = load_composed_config(
        CONFIGS / "posttrain" / "robotwin2.yaml"
    )
    assert default_config.batch_adapter.diffusion_view_layout == "mosaic_2x2"
    assert (
        default_config.model.model_config.model.visual_memory.diffusion_view_layout
        == "mosaic_2x2"
    )

    monkeypatch.setenv("WSP_DIFFUSION_VIEW_LAYOUT", "robotwin_concat")
    monkeypatch.setenv("WAM_FRAME_SEQLEN", "120")
    config = load_composed_config(CONFIGS / "posttrain" / "robotwin2.yaml")

    assert config.batch_adapter.diffusion_view_layout == "robotwin_concat"
    assert config.model.model_config.model.wam.core.parameters.frame_seqlen == 120
    assert (
        config.model.model_config.model.visual_memory.diffusion_view_layout
        == "robotwin_concat"
    )

    model_config = OmegaConf.to_container(
        config.model.model_config,
        resolve=True,
    )
    del model_config["model"]["visual_memory"]["diffusion_view_layout"]
    parsed = ModelConfig.from_dict(model_config)
    assert parsed.model.visual_memory.diffusion_view_layout == "mosaic_2x2"


def test_modular_config_layout_has_no_phase_duplicates_or_orphans():
    assert not (CONFIGS / "policy").exists()
    assert {path.name for path in (CONFIGS / "model").glob("*.yaml")} == {
        "wsp2_wan22_5b.yaml"
    }
    assert {path.name for path in (CONFIGS / "conditioning").glob("*.yaml")} == {
        "interactive_t5.yaml",
        "auto_qwen3vl.yaml",
    }
    assert {path.name for path in (CONFIGS / "memory").glob("*.yaml")} == {
        "event_default.yaml",
        "visual_prefill_default.yaml",
    }
    assert {path.name for path in (CONFIGS / "wam").glob("*.yaml")} == {
        "wan22_5b.yaml"
    }
    assert not list(CONFIGS.rglob("*_posttrain.yaml"))


def test_two_stage_mixed_pretrain_contracts(monkeypatch):
    for key, value in {
        "PRETRAINED_MODEL_PATH": "/stage1",
        "TOKENIZER_DIR": "/tokenizer",
        "WAN_CKPT_DIR": "/wan",
        "CLIP_CKPT_DIR": "/clip",
        "Qwen_CKPT_DIR": "/qwen",
        "T2VA_DATA_ROOT": "/text",
        "GOAL_IMAGE_DATA_ROOT": "/goal",
        "VIDEO_DATA_ROOT": "/video",
    }.items():
        monkeypatch.setenv(key, value)
    stage1 = resolve_config_profiles(
        load_composed_config(
            CONFIGS / "pretrain" / "mixed_three_mode_wan22.yaml"
        )
    )
    stage2 = resolve_config_profiles(
        load_composed_config(
            CONFIGS / "pretrain" / "mixed_three_mode_wan22_stage2.yaml"
        )
    )

    assert stage1.data_loader.dataset_name == "worldscape_hdf5_mixed_pretrain"
    assert stage2.data_loader.dataset_name == stage1.data_loader.dataset_name
    assert stage1.model.model_config.model.mode == "interactive"
    assert stage1.model.model_config.model.event_memory.enabled is False
    assert stage1.data_loader.action_mode == "eef"
    assert stage1.data_loader.relative_action is True
    assert stage1.prompt_schedule.enabled is False
    assert stage1.selectors.deepspeed_mode == "zero2"
    assert stage1.distributed.backend == "deepspeed"
    assert stage2.model.initialization == "checkpoint_overlay"
    assert stage2.model.model_config.model.mode == "auto"
    assert stage2.model.model_config.model.event_memory.enabled is True
    assert stage2.data_loader.action_mode == "eef"
    assert stage2.data_loader.relative_action is True
    assert stage2.prompt_schedule.enabled is True
    assert stage2.objective.semantic_forcing_weight == 0.001
    stage2_vlm = (
        stage2.model.model_config.model.condition_router.auto.vlm.parameters
    )
    assert stage2_vlm.vlm_token_mode == "last"
    assert "freeze_vlm" not in stage2_vlm
    assert "freeze_qformer" not in stage2_vlm
    assert stage2.freeze.config.qformer is None
    for config in (stage1, stage2):
        model_config = OmegaConf.to_container(
            config.model.model_config, resolve=True
        )
        generation_config = OmegaConf.to_container(
            config.model.generation_config, resolve=True
        )
        ModelConfig.from_dict(model_config)
        GenerationConfig.from_dict(generation_config)
        assert config.model.model_config.model.visual_memory.persistent_prompt == (
            "goal_or_demo"
        )
        geometry = config.noise_kernel.config
        assert (
            geometry.num_frame_per_block,
            geometry.num_action_per_block,
            geometry.num_state_per_block,
        ) == (2, 24, 1)


def test_stage1_checkpoint_resume_does_not_require_raw_component_env(monkeypatch):
    for key in (
        "WAN_CKPT_DIR",
        "CLIP_CKPT_DIR",
        "TOKENIZER_DIR",
        "T5_CKPT_PATH",
        "VAE_CKPT_PATH",
        "CLIP_CKPT_PATH",
        "Qwen_CKPT_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "PRETRAINED_MODEL_PATH": "/stage1",
        "WSP_INITIALIZATION": "checkpoint",
        "T2VA_DATA_ROOT": "/text",
        "GOAL_IMAGE_DATA_ROOT": "/goal",
        "VIDEO_DATA_ROOT": "/video",
    }.items():
        monkeypatch.setenv(key, value)

    stage1 = resolve_config_profiles(
        load_composed_config(
            CONFIGS / "pretrain" / "mixed_three_mode_wan22.yaml"
        )
    )

    assert stage1.model.initialization == "checkpoint"
    assert stage1.model.text_encoder_pretrained_path is None
    assert stage1.model.vae_pretrained_path is None
    assert stage1.model.image_encoder_pretrained_path is None
    assert stage1.model.diffusion_model_pretrained_path is None


def _expected_deepspeed(mode: str) -> dict:
    common = {
        "zero_allow_untested_optimizer": True,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
    }
    if mode == "zero2":
        zero = {
            "stage": 2,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "sub_group_size": 1.0e9,
            "reduce_bucket_size": 1.0e8,
        }
    elif mode == "zero2_offload":
        common["zero_force_ds_cpu_optimizer"] = False
        zero = {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": False,
            "contiguous_gradients": True,
            "sub_group_size": 1.0e9,
            "reduce_bucket_size": 1.0e8,
        }
    else:
        zero = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1.0e9,
            "reduce_bucket_size": 5.0e8,
            "stage3_prefetch_bucket_size": 5.0e8,
            "stage3_param_persistence_threshold": 1.0e6,
            "stage3_max_live_parameters": 1.0e9,
            "stage3_max_reuse_distance": 1.0e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        }
    return {**common, "zero_optimization": zero}


def test_all_48_posttrain_selector_combinations_preserve_exact_routing(monkeypatch):
    monkeypatch.delenv("PRETRAINED_MODEL_FORMAT", raising=False)
    environment = {
        "PRETRAINED_MODEL_PATH": "/policy",
        "TOKENIZER_DIR": "/tokenizer",
        "WAN_CKPT_DIR": "/wan",
        "CLIP_CKPT_DIR": "/clip",
        "Qwen_CKPT_DIR": "/qwen",
        "DATA_ROOT": "/data",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    platform_datasets = {
        "agilex": (
            ("none", "worldscape_hdf5_text"),
            ("goal", "worldscape_hdf5_goal"),
            ("demo", "worldscape_hdf5_demo"),
        ),
        "libero": (
            ("none", "worldscape_lerobot_text"),
            ("goal", "worldscape_lerobot_goal"),
            ("demo", "worldscape_lerobot_demo"),
        ),
        "robotwin2": (
            ("none", "worldscape_lerobot_text"),
            ("goal", "worldscape_lerobot_goal"),
            ("demo", "worldscape_lerobot_demo"),
        ),
    }
    visual_semantics = {
        "none": ("none", "none", 1, False),
        "goal": ("last", "last", 1, True),
        "demo": ("uniform", "uniform", 50, True),
    }
    combinations = [
        (platform, mode, prompt, dataset, deepspeed)
        for platform, pairs in platform_datasets.items()
        for mode, (prompt, dataset), deepspeed in product(
            ("interactive", "auto"),
            pairs,
            ("zero2", "zero2_offload", "zero3"),
        )
    ]
    assert len(combinations) == 54

    for platform, mode, prompt, dataset, deepspeed in combinations:
        config = resolve_config_profiles(
            load_composed_config(CONFIGS / "posttrain" / f"{platform}.yaml"),
            overrides=OmegaConf.from_dotlist(
                [
                    f"selectors.mode={mode}",
                    f"selectors.visual_prompt={prompt}",
                    f"selectors.dataset_name={dataset}",
                    f"selectors.deepspeed_mode={deepspeed}",
                ]
            ),
        )
        ModelConfig.from_dict(
            OmegaConf.to_container(config.model.model_config, resolve=True)
        )
        GenerationConfig.from_dict(
            OmegaConf.to_container(
                config.model.generation_config, resolve=True
            )
        )
        _, sampling, context_len, head_only = visual_semantics[prompt]
        assert config.model.model_config.model.mode == mode
        assert config.model.model_config.model.event_memory.enabled is (
            mode == "auto"
        )
        assert config.selectors.deepspeed_mode == deepspeed
        assert config.model.expected_mode == mode
        assert config.model.tokenizer_path == "/tokenizer"
        assert config.model.text_encoder_pretrained_path == (
            "/wan/models_t5_umt5-xxl-enc-bf16.pth"
        )
        assert config.data_loader.mode == mode
        assert config.data_loader.dataset_name == dataset
        assert config.data_loader.action_mode == (
            "joint" if platform == "robotwin2" else "eef"
        )
        assert config.data_loader.relative_action is False
        kwargs = config.data_loader.dataset_kwargs
        assert (
            kwargs.visual_prompt,
            kwargs.context_sampling_mode,
            kwargs.context_video_len,
            kwargs.ctx_head_only,
        ) == (prompt, sampling, context_len, head_only)
        assert config.freeze.config.qformer is None
        assert config.freeze.config.event_memory is (mode == "interactive")
        assert config.objective.semantic_forcing_weight == (
            0.001 if mode == "auto" else 0.0
        )
        assert config.prompt_schedule.enabled is (mode == "auto")
        assert OmegaConf.to_container(
            config.distributed.deepspeed_config, resolve=True
        ) == _expected_deepspeed(deepspeed)

        if mode == "interactive":
            assert config.model.initialization == "auto"
            assert "legacy_action_overrides" not in config.model
            assert config.model.model_config.model.mode == "interactive"
        else:
            vlm = config.model.model_config.model.condition_router.auto.vlm.parameters
            projector = config.model.model_config.model.condition_router.auto.projector
            assert vlm.vlm_token_mode == "last"
            assert "freeze_vlm" not in vlm
            assert "freeze_qformer" not in vlm
            assert vlm.qformer_output_dim == config.model.model_config.model.shape.vlm_token_dim
            assert projector.input_dim == vlm.qformer_output_dim
            assert vlm.enable_planning_branch is True
            assert config.model.diffusion_model_pretrained_path == "/wan"
            assert config.model.image_encoder_pretrained_path == (
                "/clip/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            )
            assert config.model.vae_pretrained_path == "/wan/Wan2.2_VAE.pth"
            assert config.model.vlm_pretrained_path == "/qwen"
            assert config.model.initialization == "checkpoint_overlay"
            assert config.model.model_config.model.mode == "auto"
            assert config.model.model_config.model.event_memory.enabled is True
            assert "legacy_action_overrides" not in config.model
