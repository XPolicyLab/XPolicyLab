"""Real tiny joint-v2 full-forward sampling, no GPU/checkpoint dependency."""

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from wall_x.model.vap.model import VAPTokenModel
from wall_x.model.vap.training import VAPTrainingModel
from wall_x.model.vap.wan_bridge import EncodedFirstFrameMemory
from wall_x.model.vapv2.model import VAPv2TokenModel
from wall_x.model.vapv2.text import JointTextProjector, TextCondition
from wall_x.model.vapv2.inference import FullForwardSamplingModel, _BoundModel
from wall_x.model.vapv2.serving import _verify_online_qwen_matches_cache
from wall_x.model.vap import sampling
from scripts.vapv2.infer_robodojo import prepare_memory, infer_window, parser


def test_online_qwen_from_cache_requires_exact_source_files(tmp_path):
    model = tmp_path / "qwen"
    model.mkdir()
    files = {
        "config.json": b"config",
        "model.safetensors": b"weights",
        "tokenizer.json": b"tokens",
    }
    for name, contents in files.items():
        (model / name).write_bytes(contents)
    metadata = {
        "source_files_sha256": {
            name: hashlib.sha256(contents).hexdigest()
            for name, contents in files.items()
        }
    }
    _verify_online_qwen_matches_cache(model, metadata)
    (model / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="differs from training cache"):
        _verify_online_qwen_matches_cache(model, metadata)
    (model / "model.safetensors").write_bytes(files["model.safetensors"])
    (model / "extra.json").write_bytes(b"different config")
    with pytest.raises(ValueError, match="file inventory"):
        _verify_online_qwen_matches_cache(model, metadata)


def tiny():
    torch.manual_seed(42)
    core = VAPv2TokenModel.from_existing(
        VAPTokenModel(
            video_dim=8,
            action_dim=6,
            action_input_dim=4,
            video_output_dim=8,
            num_layers=2,
            num_heads=1,
            head_dim=8,
            video_ffn_dim=16,
            action_ffn_dim=12,
            num_chunks=2,
            rope_axes_dim=(2, 2, 2, 2),
            frequency_dim=8,
            attention_backend="sdpa",
        )
    )
    core.text_input_projector = JointTextProjector(6, 8)
    core.video_patch_embedding = nn.Conv3d(8, 8, 1)
    core.video_patch_size = (1, 1, 1)
    nn.init.normal_(core.action_head.head.weight, std=0.1)
    config = SimpleNamespace(
        num_interleave_chunks=2,
        video_frames_per_chunk=2,
        vae_temporal_stride=1,
        action_steps_per_chunk=2,
        text_feature_seq_len=4,
        text_feature_hidden_dim=6,
        video_flow=dict(shift=1.0, train_timesteps=1000, inference_steps=2),
        action_flow=dict(shift=1.0, train_timesteps=1000, inference_steps=3),
        first_frame_memory_enabled=True,
        first_frame_memory_grid=(1, 1),
        first_frame_memory_injection_mode="prefix_self_attention",
        first_frame_memory_position_mode="relative_age",
        history_offsets=(2, 4, 6),
    )
    model = VAPTrainingModel(
        vae=nn.Identity(),
        core=core,
        config=config,
        prompter=None,
        torch_dtype=torch.float32,
    ).eval()
    text = TextCondition(
        torch.randn(1, 4, 6), torch.tensor([[True, True, False, False]])
    )
    memory = EncodedFirstFrameMemory(
        latents=torch.randn(1, 1, 8, 4, 1, 1),
        view_keys=("view1",),
        valid=torch.tensor([[True, False, False, True]]),
        age_steps=torch.tensor([[5, 0, 0, 2]]),
    )
    kwargs = dict(
        condition_video_latents=torch.randn(1, 1, 8, 1, 1, 1),
        view_ids=torch.tensor([[1]]),
        first_frame_memory=memory,
        video_noise=torch.randn(1, 1, 8, 4, 1, 1),
        action_noise=torch.randn(1, 4, 4),
    )
    return model, text, kwargs


@pytest.mark.parametrize("stage", ["a0", "a1"])
def test_real_full_forward_two_stages_and_request_mask(stage):
    model, text, kwargs = tiny()
    facade = FullForwardSamplingModel(model)
    method = facade.sample_first_action_chunk_given_video
    fn = sampling.sample_first_action_chunk_given_video
    if stage == "a1":
        kwargs.update(
            clean_prefix_video_latents=torch.randn(1, 1, 8, 2, 1, 1),
            clean_prefix_action=torch.randn(1, 2, 4),
        )
        method = facade.sample_next_action_chunk_given_prefix
        fn = sampling.sample_next_action_chunk_given_prefix
    seen = []
    model.core.register_forward_pre_hook(
        lambda m, a, k: seen.append(k), with_kwargs=True
    )
    expected = fn(_BoundModel(model, text), text_context=text.hidden_states, **kwargs)
    actual = method(text_context=text, **kwargs)
    torch.testing.assert_close(actual.generated_chunk, expected.generated_chunk)
    assert actual.generated_chunk.shape == (1, 2, 4)
    assert torch.isfinite(actual.generated_chunk).all()
    assert len(seen) == 6  # three full joint forwards per sample, no cached decode
    assert all(torch.equal(k["text_valid_mask"], text.valid_mask) for k in seen)
    assert all(
        k["memory_prefix_valid"].tolist() == [[True, False, False, True]] for k in seen
    )
    assert not hasattr(model.core, "_v2_text_valid_mask")
    # Inactive history cannot affect actions, including A1 with its clean prefix.
    memory = kwargs["first_frame_memory"]
    memory.latents[:, :, :, 1:3] += 1000
    perturbed = TextCondition(text.hidden_states.clone(), text.valid_mask)
    perturbed.hidden_states[:, 2:] = float("nan")
    again = method(text_context=perturbed, **kwargs)
    torch.testing.assert_close(actual.generated_chunk, again.generated_chunk)
    # A second prompt uses its own validity despite the same model instance.
    other = TextCondition(text.hidden_states, torch.tensor([[True, True, True, False]]))
    method(text_context=other, **kwargs)
    assert torch.equal(seen[-1]["text_valid_mask"], other.valid_mask)
    assert torch.equal(seen[0]["text_valid_mask"], text.valid_mask)


def test_sampling_rejects_legacy_cache_tensor_and_train_mode():
    model, text, kwargs = tiny()
    sample = FullForwardSamplingModel(model).sample_first_action_chunk_given_video
    model.core.state_film = nn.Identity()
    with pytest.raises(NotImplementedError, match="State-FiLM"):
        sample(text_context=text, use_kv_cache=True, **kwargs)
    model.core.state_film = None
    with pytest.raises(TypeError, match="TextCondition"):
        sample(text_context=text.hidden_states, **kwargs)
    invalid = TextCondition(
        text.hidden_states, torch.tensor([[True, False, True, False]])
    )
    with pytest.raises(ValueError, match="right padding"):
        sample(text_context=invalid, **kwargs)
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        sample(text_context=text, **kwargs)


def fake_runtime():
    runtime = SimpleNamespace(
        first_frame_memory_enabled=True,
        history_offsets=(32, 64, 96),
        first_frame_memory_position_mode="relative_age",
        video_frames_per_chunk=16,
        state_conditioning_enabled=False,
        encode_first_frame_memory=Mock(side_effect=lambda x: int(x[0, 0, 0, 0, 0])),
        assemble_history_memory=Mock(return_value="bank"),
        encode_text=Mock(return_value="text"),
        sample_first_action=Mock(return_value=torch.zeros(32, 20)),
        sample_next_action=Mock(return_value=torch.zeros(32, 20)),
        prepare_action_for_execution=Mock(
            return_value=(np.zeros((32, 20)), np.ones((32, 20)))
        ),
    )
    return runtime


@pytest.mark.parametrize(
    "step,steps,expected",
    [
        (0, [0], [True, False, False, False]),
        (64, [0, 32], [True, False, False, True]),
        (128, [0, 32, 64, 96], [True, True, True, True]),
        (128, [0, 64], [True, False, True, False]),
    ],
)
def test_memory_selection_and_missing_slots(step, steps, expected):
    runtime = fake_runtime()
    payload = dict(
        history_steps=np.array(steps),
        history_rgb=np.stack([np.full((3, 2, 2, 3), s, dtype=np.uint8) for s in steps]),
    )
    assert prepare_memory(runtime, payload, window_start_step=step) == ("bank", None)
    anchors, valid, ages = runtime.assemble_history_memory.call_args.args
    assert valid == expected
    assert ages[0] == step
    assert all(age == 0 for age, ok in zip(ages, valid) if not ok)
    assert runtime.encode_first_frame_memory.call_count == sum(valid)


def test_entry_keeps_normalized_and_physical_outputs_distinct():
    runtime = fake_runtime()
    payload = dict(
        video=np.zeros((3, 1, 2, 2, 3), dtype=np.uint8),
        history_steps=np.array([0]),
        history_rgb=np.zeros((1, 3, 2, 2, 3), dtype=np.uint8),
    )
    result = infer_window(
        runtime, payload, stage="a0", instruction="pick", window_start_step=0, seed=3
    )
    assert not result["normalized_action"].any()
    assert result["relative_action"].all()
    assert not runtime.sample_next_action.called
    payload["video"] = np.zeros((3, 17, 2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="previous_normalized_action"):
        infer_window(
            runtime,
            payload,
            stage="a1",
            instruction="pick",
            window_start_step=0,
            seed=3,
        )
    payload["previous_normalized_action"] = np.zeros((32, 20))
    infer_window(
        runtime, payload, stage="a1", instruction="pick", window_start_step=0, seed=3
    )
    assert runtime.sample_next_action.called
    assert "--use-kv-cache" not in parser().format_help()


def test_memory_rejects_future_and_missing_anchor():
    runtime = fake_runtime()
    payload = dict(
        history_steps=np.array([0, 32]),
        history_rgb=np.zeros((2, 3, 2, 2, 3), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="newer"):
        prepare_memory(runtime, payload, window_start_step=0)
    payload = dict(
        history_steps=np.array([32]),
        history_rgb=np.zeros((1, 3, 2, 2, 3), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="first frame"):
        prepare_memory(runtime, payload, window_start_step=64)


@pytest.mark.parametrize("remove_key", [False, True])
def test_loader_strict_weights_no_training_initializer_dependency(
    tmp_path, monkeypatch, remove_key
):
    from safetensors.torch import save_file
    from wall_x.model.vapv2 import serving as module
    from wall_x.model.vapv2.training import VAPv2TrainingModel
    from wall_x.model.vap import backbone
    from wall_x.config import loader

    model, _, _ = tiny()
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    if remove_key:
        state.pop("core.text_input_projector.modality")
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.yml").write_text("checkpoint-owned")
    cfg = SimpleNamespace(
        model_type="vapv2",
        action_dim=4,
        propri_dim=0,
        data=SimpleNamespace(dataset_type="x2robot_v2"),
        model=SimpleNamespace(
            architecture_version="vapv2.qwen_joint_ivava.v2", state_film_enabled=False
        ),
        checkpoint=SimpleNamespace(resume_from="/does/not/exist/pretrain"),
    )
    monkeypatch.setattr(loader, "load_config", lambda path: cfg)
    monkeypatch.setattr(module, "_build_action_normalizer", lambda *a: (Mock(), "test"))
    monkeypatch.setattr(
        module, "_validate_serving_normalizer_contract", lambda *a: None
    )
    monkeypatch.setattr(backbone, "load_wan22_5b_video_base", lambda *a, **k: object())
    monkeypatch.setattr(module, "build_vapv2_model_structure", lambda *a, **k: model)
    monkeypatch.setattr(VAPv2TrainingModel, "from_wan_pipeline", lambda *a, **k: model)

    class Capture(module.VAPv2ServingRuntime):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    if remove_key:
        with pytest.raises(RuntimeError, match="Missing key"):
            Capture.from_paths(checkpoint_path=tmp_path, device="cpu")
    else:
        runtime = Capture.from_paths(checkpoint_path=tmp_path, device="cpu")
        assert runtime.kwargs["model"] is model
        assert not model.training
    with pytest.raises(ValueError, match="own config"):
        Capture.from_paths(checkpoint_path=tmp_path, config_path=tmp_path / "other.yml")


def test_encode_text_uses_explicit_mask_not_training_side_channel(monkeypatch):
    from wall_x.model.vapv2.serving import VAPv2ServingRuntime

    model, _, _ = tiny()
    model.config.prompt_template = None
    model.config.prompt_priority_order = None
    model.prompter = SimpleNamespace(lengths={"pick": 2})
    hidden = torch.randn(1, 4, 6)
    monkeypatch.setattr(
        VAPTrainingModel, "_encode_text_context", lambda *a, **k: hidden
    )
    runtime = object.__new__(VAPv2ServingRuntime)
    runtime.model = FullForwardSamplingModel(model)
    runtime.device = torch.device("cpu")
    text = runtime.encode_text("pick")
    assert text.valid_mask.tolist() == [[True, True, False, False]]
    assert not hasattr(model.core, "_v2_text_valid_mask")
