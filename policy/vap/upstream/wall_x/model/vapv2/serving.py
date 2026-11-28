"""Independent joint-v2 RoboDojo runtime; legacy VAP serving is unchanged.

RGB input [3,T,H,W,3], action output [A,20] in the checkpoint's normalized
coordinates. T=1 for A0; T=1+V for A1 with an executed clean A0 prefix.
History images are individually VAE-encoded, never temporally compressed together.
"""

from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path

import torch

from wall_x.model.vap.serving import (
    VAPServingRuntime,
    _build_action_normalizer,
    _build_x2robot_v2_proprio_normalizer,
    _load_matching_normalizer_state,
    _resolve_checkpoint_file,
    _validate_serving_normalizer_contract,
)
from .inference import FullForwardSamplingModel
from .structure import build_vapv2_model_structure
from .text import TextCondition


def _restore_serving_model(cfg, checkpoint, requested_device, *, structure_builder):
    """Restore every persistent VAP/VAE tensor; scaffolding is not a weight source."""
    from safetensors.torch import load_file

    builder = structure_builder or build_vapv2_model_structure
    model = builder(
        cfg.model,
        action_dim=cfg.action_dim,
        state_dim=cfg.propri_dim,
    )
    state = load_file(str(checkpoint), device="cpu")
    # Preserve only the small output head before load_state_dict casts the
    # whole checkpoint to the trunk's BF16 dtype. The opt-in FP32 mode must
    # use original checkpoint values; legacy still follows the old cast.
    model.core._v2_action_head_checkpoint_fp32 = {
        name: state[f"core.action_head.{name}"].detach().float().clone()
        for name in model.core.action_head.state_dict()
    }
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    model.to(device=requested_device).eval()
    return model


def _verify_online_qwen_matches_cache(model_path: Path, cache_metadata: dict) -> None:
    """Require the exact weights/tokenizer used to build the training cache."""
    expected = cache_metadata.get("source_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Qwen training cache has no source file hashes")
    source = Path(model_path).expanduser().resolve()
    actual_files = {
        p.name
        for p in source.iterdir()
        if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt"}
    }
    if actual_files != set(expected):
        raise ValueError(
            "Online Qwen file inventory differs from training cache provenance"
        )
    for name, digest in expected.items():
        path = source / name
        observed = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                observed.update(block)
        if observed.hexdigest() != digest:
            raise ValueError(f"Online Qwen file differs from training cache: {name}")


class VAPv2ServingRuntime(VAPServingRuntime):
    def __init__(self, *, model, use_kv_cache=False, action_acceleration=None, **kwargs):
        if kwargs["cfg"].model_type != "vapv2":
            raise ValueError("VAPv2 runtime requires model_type=vapv2")
        if use_kv_cache and model.core.state_film is not None:
            raise NotImplementedError(
                "VAPv2 prefix cache does not support State-FiLM"
            )
        super().__init__(
            model=FullForwardSamplingModel(model), use_kv_cache=use_kv_cache, **kwargs
        )
        import os
        from .action_precision import configure_action_precision
        precision = os.environ.get("VAP_V2_ACTION_PRECISION", "legacy")
        precision_metadata = configure_action_precision(
            model.core, precision,
            initial_noise_dtype=os.environ.get("VAP_V2_ACTION_NOISE_DTYPE", "legacy"),
        )
        print(f"[VAPv2 action precision] {precision_metadata}", flush=True)
        mode = (os.environ.get("VAP_V2_ACTION_ACCELERATION", "off")
                if action_acceleration is None else action_acceleration)
        if mode not in {"off", "rope", "graph_rope"}:
            raise ValueError("VAP_V2_ACTION_ACCELERATION must be off/rope/graph_rope")
        if mode != "off":
            if not use_kv_cache:
                raise ValueError("V2 action acceleration requires KV-cache")
            if os.environ.get("VAP_FUSED_ROPE", "0") != "0":
                raise ValueError("Do not combine V2 inference RoPE with training fused RoPE")
            from .inference_acceleration import VAPv2ActionAcceleration
            accelerator = VAPv2ActionAcceleration(model.core, mode=mode)
            print(f"[VAPv2 acceleration] {accelerator.metadata()}", flush=True)

    @classmethod
    def from_paths(
        cls,
        *,
        checkpoint_path,
        config_path=None,
        device="cuda:0",
        normalizer_key=None,
        normalizer_stats_path=None,
        text_feature_cache_path=None,
        qwen_model_path=None,
        qwen_online_from_cache=False,
        view_ids=(1, 0, 2),
        action_inference_steps=None,
        action_inference_shift=None,
        use_kv_cache=False,
        action_acceleration=None,
        vae_tiled=False,
        structure_builder=None,
    ):
        """Strict weights + checkpoint-owned config; overrides relocate assets only.

        Missing normalizer .pth is allowed for two-file inference exports. JSON
        statistics remain mandatory; if .pth exists it must match those stats.
        No optimizer, source initialization checkpoint, or dataset is loaded.
        """
        checkpoint = _resolve_checkpoint_file(checkpoint_path)
        own_config = checkpoint.parent / "config.yml"
        if (
            config_path is not None
            and Path(config_path).expanduser().resolve() != own_config
        ):
            raise ValueError(
                "Use the checkpoint own config.yml; relocate assets with explicit overrides"
            )
        from wall_x.config.loader import load_config

        cfg = load_config(str(own_config))
        if use_kv_cache and cfg.model.state_film_enabled:
            raise NotImplementedError(
                "VAPv2 prefix cache does not support State-FiLM"
            )
        if (
            cfg.model_type != "vapv2"
            or cfg.model.architecture_version != "vapv2.qwen_joint_ivava.v2"
        ):
            raise ValueError(
                "Expected a joint-v2 VAPv2 checkpoint, not legacy VAP/hybrid-v1"
            )
        if cfg.data.dataset_type != "x2robot_v2":
            raise ValueError(
                "This RoboDojo entry requires x2robot_v2 action normalization"
            )
        if normalizer_stats_path is not None:
            stats = Path(normalizer_stats_path).expanduser().resolve()
            if not stats.is_file():
                raise FileNotFoundError(stats)
            cfg.data.normalizer_config = dict(cfg.data.normalizer_config)
            cfg.data.normalizer_config["customized_action_statistic_dof"] = str(stats)
        if qwen_online_from_cache:
            if (
                cfg.model.text_encoder_backend != "qwen35_cache"
                or qwen_model_path is None
            ):
                raise ValueError(
                    "Online-from-cache requires a Qwen cache checkpoint and --qwen-model"
                )
            from wall_x.model.vap.text_conditioning import Qwen35TextCache
            from wall_x.model.vap.qwen35_online import model_identity

            cache_path = (
                Path(text_feature_cache_path or cfg.model.text_feature_cache_path)
                .expanduser()
                .resolve()
            )
            cache = Qwen35TextCache(
                cache_path,
                expected_id=cfg.model.text_feature_cache_id,
                hidden_dim=cfg.model.text_feature_hidden_dim,
                seq_len=cfg.model.text_feature_seq_len,
            )
            model_path = Path(qwen_model_path).expanduser().resolve()
            _verify_online_qwen_matches_cache(model_path, cache.metadata)
            identity, width = model_identity(model_path)
            if width != cfg.model.text_feature_hidden_dim:
                raise ValueError("Online Qwen hidden width differs from training cache")
            if cache.metadata.get("recipe") != cfg.model.qwen35_text_recipe:
                raise ValueError("Online Qwen recipe differs from training cache")
            if cache.metadata.get("prompt_template") != cfg.model.prompt_template:
                raise ValueError(
                    "Online Qwen prompt template differs from training cache"
                )
            import transformers

            if cache.metadata.get("transformers_version") != transformers.__version__:
                raise ValueError(
                    "Online Qwen Transformers version differs from training cache"
                )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            if tokenizer.eos_token_id != cache.metadata.get("eos_token_id"):
                raise ValueError("Online Qwen EOS token differs from training cache")
            cfg.model.text_encoder_backend = "qwen35_online"
            cfg.model.qwen35_model_path = str(model_path)
            cfg.model.qwen35_model_id = identity
            cfg.model.text_feature_cache_path = None
            cfg.model.text_feature_cache_id = None
            del cache
        elif text_feature_cache_path is not None:
            if cfg.model.text_encoder_backend != "qwen35_cache":
                raise ValueError(
                    "Cache relocation requires a cache-backed checkpoint; backend conversion is not implicit"
                )
            cfg.model.text_feature_cache_path = str(
                Path(text_feature_cache_path).expanduser().resolve()
            )
        if qwen_model_path is not None and not qwen_online_from_cache:
            if cfg.model.text_encoder_backend != "qwen35_online":
                raise ValueError("Qwen model relocation requires an online checkpoint")
            from wall_x.model.vap.qwen35_online import model_identity

            path = Path(qwen_model_path).expanduser().resolve()
            identity, width = model_identity(path)
            if (
                identity != cfg.model.qwen35_model_id
                or width != cfg.model.text_feature_hidden_dim
            ):
                raise ValueError(
                    "Relocated Qwen weights/tokenizer identity differs from checkpoint"
                )
            cfg.model.qwen35_model_path = str(path)
        if action_inference_steps is not None:
            if int(action_inference_steps) <= 0:
                raise ValueError("action_inference_steps must be positive")
            cfg.model.action_flow["inference_steps"] = int(action_inference_steps)
        if action_inference_shift is not None:
            if (
                not math.isfinite(float(action_inference_shift))
                or float(action_inference_shift) <= 0
            ):
                raise ValueError("action_inference_shift must be positive and finite")
            cfg.model.action_flow["shift"] = float(action_inference_shift)
        normalizer, key = _build_action_normalizer(cfg, normalizer_key)
        norm_state = checkpoint.parent / "normalizer_action.pth"
        if norm_state.is_file():
            _load_matching_normalizer_state(
                normalizer, norm_state, label=norm_state.name
            )
        _validate_serving_normalizer_contract(cfg, normalizer, key)
        proprio = None
        if cfg.model.state_film_enabled:
            proprio = _build_x2robot_v2_proprio_normalizer(
                cfg,
                stats_path=cfg.data.normalizer_config[
                    "customized_action_statistic_dof"
                ],
                normalizer_key=key,
            )
            state_file = checkpoint.parent / "normalizer_propri.pth"
            if state_file.is_file():
                _load_matching_normalizer_state(
                    proprio, state_file, label=state_file.name
                )
        requested_device = torch.device(device)
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_device(requested_device)
        model = _restore_serving_model(
            cfg, checkpoint, requested_device,
            structure_builder=structure_builder,
        )
        normalizer.to("cpu")
        if proprio is not None:
            proprio.to("cpu")
        return cls(
            cfg=cfg,
            model=model,
            normalizer_action=normalizer,
            normalizer_propri=proprio,
            normalizer_key=key,
            device=requested_device,
            view_ids=view_ids,
            vae_tiled=vae_tiled,
            use_kv_cache=use_kv_cache,
            action_acceleration=action_acceleration,
        )

    @torch.no_grad()
    def encode_text(self, instruction: str) -> TextCondition:
        from wall_x.model.vap.training import VAPTrainingModel, build_prompt_from_data

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("VAPv2 requires a nonempty instruction")
        batch = {"instruction": [instruction.strip()]}
        model = self.model._model
        # Base implementation encodes raw features without mutating v2's
        # training-only _v2_text_valid_mask side channel.
        hidden = VAPTrainingModel._encode_text_context(
            model,
            batch,
            batch_size=1,
            device=self.device,
            dtype=self.dtype,
        )
        prompt = build_prompt_from_data(
            batch,
            use_qwen25=False,
            prompt_template=model.config.prompt_template,
            qwen25_prompt_template=None,
            prompt_priority_order=model.config.prompt_priority_order,
        )
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        provider = model.prompter
        if hasattr(provider, "lengths"):
            lengths = [provider.lengths[p] for p in prompts]
        elif getattr(provider, "tokenizer", None) is not None:
            from wall_x.model.vap.text_conditioning import (
                qwen35_encode_ids, qwen35_truncation_for_recipe,
            )

            lengths = [
                len(qwen35_encode_ids(
                    provider.tokenizer, p,
                    seq_len=model.config.text_feature_seq_len,
                    eos_token_id=provider.tokenizer.eos_token_id,
                    truncation=getattr(
                        model.config, "qwen35_truncation",
                        qwen35_truncation_for_recipe(model.config.qwen35_text_recipe),
                    ),
                ))
                for p in prompts
            ]
        else:
            raise ValueError(
                "Qwen provider must expose cache lengths or an EOS tokenizer"
            )
        if any(not 0 < n <= hidden.shape[1] for n in lengths):
            raise ValueError("Qwen token length differs from the configured encoding recipe")
        valid = (
            torch.arange(hidden.shape[1], device=hidden.device)[None]
            < torch.tensor(lengths, device=hidden.device)[:, None]
        )
        return TextCondition(hidden.detach(), valid)

    def encode_video(self, video):
        pixels = torch.as_tensor(video)
        expected = (int(self.cfg.model.height), int(self.cfg.model.width), 3)
        if pixels.ndim != 5 or tuple(pixels.shape[-3:]) != expected:
            raise ValueError(
                f"Supply preprocessed RGB [3,T,{expected[0]},{expected[1]},3]"
            )
        if pixels.dtype != torch.uint8 and not pixels.is_floating_point():
            raise TypeError("RGB must be uint8 [0,255] or floating [0,1]")
        if pixels.is_floating_point() and (
            not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1
        ):
            raise ValueError("Floating RGB must be finite in [0,1]")
        # Convert uint8 explicitly so a nearly black image with max=1 is not
        # mistaken by the shared pixel helper for already normalized floats.
        return super().encode_video(
            pixels.float() / 255 if pixels.dtype == torch.uint8 else pixels
        )
