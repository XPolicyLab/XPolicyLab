"""VAPv2 serving must construct structure without opening Wan weight files."""
from types import SimpleNamespace as NS

import pytest
import torch

from wall_x.model.vapv2.structure import build_vapv2_model_structure


def config():
    return NS(
        architecture_version="vapv2.qwen_joint_ivava.v2",
        text_injection_mode="joint_ivava_prefix",
        base_model_id="Wan-AI/Wan2.2-TI2V-5B",
        text_encoder_backend="qwen35_online",
        qwen35_model_path="/does/not/read/qwen/during/structure",
        qwen35_model_id="immutable-test-identity",
        qwen35_text_recipe="plain_text_head_truncate_eos_last_hidden_v1",
        qwen35_truncation="head",
        qwen35_encode_micro_batch=1,
        text_feature_seq_len=512,
        text_feature_hidden_dim=4096,
        text_feature_cache_path=None,
        text_feature_cache_id=None,
        text_projector_output_dim=3072,
        text_feature_cache_online_fallback=False,
        action_expert_dim=1024,
        num_interleave_chunks=2,
        rope_axes_dim=(8, 36, 42, 42),
        max_views=64,
        joint_num_heads=24,
        joint_head_dim=128,
        attention_backend="flex_attention",
        state_film_enabled=False,
        first_frame_memory_enabled=True,
        first_frame_memory_injection_mode="prefix_self_attention",
        first_frame_memory_reader_layers=(3, 9, 15, 21, 27),
        video_flow={"shift": 5.0, "sample_mode": "uniform"},
    )


def test_structure_has_no_wan_weight_dependency(monkeypatch):
    import wall_x.model.vap.backbone as backbone
    from wall_x.model.vga.modeling.wan_video import WanVideoPipeline

    def fail(*args, **kwargs):
        raise AssertionError("Wan weight loader must not be called")

    monkeypatch.delenv("WALLX_MODEL_ROOT", raising=False)
    monkeypatch.setattr(backbone, "load_wan22_5b_video_base", fail)
    monkeypatch.setattr(WanVideoPipeline, "from_pretrained", fail)
    torch.manual_seed(9182)
    rng_before = torch.random.get_rng_state()
    model = build_vapv2_model_structure(config(), action_dim=20, state_dim=0)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert model.core.video_dim == 3072
    assert len(model.core.blocks) == 30
    assert model.core.first_frame_memory is None
    assert model.vae.z_dim == 48
    assert model.vae.upsampling_factor == 16
    for buffer in model.vae.buffers():
        assert buffer.device.type == "cpu"
        assert buffer.dtype == torch.bfloat16
        assert torch.isfinite(buffer).all()
    for scale in model.vae.scale:
        assert scale.device.type == "cpu"
        assert scale.dtype == torch.float32
    for parameter in model.vae.parameters():
        assert parameter.dtype == torch.bfloat16
    assert model.prompter.text_encoder is None


def test_structure_rejects_non_online_or_other_base(monkeypatch):
    cfg = config()
    cfg.text_encoder_backend = "qwen35_cache"
    with pytest.raises(ValueError, match="qwen35_online"):
        build_vapv2_model_structure(cfg, action_dim=20, state_dim=0)
    cfg = config()
    cfg.base_model_id = "other"
    with pytest.raises(ValueError, match="Unsupported"):
        build_vapv2_model_structure(cfg, action_dim=20, state_dim=0)
