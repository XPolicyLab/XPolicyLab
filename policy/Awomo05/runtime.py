# Copyright (c) 2026 Awomo-WAM Team. Apache-2.0.
"""RoboDojo inference using the published Awomo checkpoint format."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)
ACTIVE = (0, 1, 2, 3, 4, 5, 16, 29, 30, 31, 32, 33, 34, 45)


def bundled_module(name):
    root = Path(__file__).resolve().parent / "awomo"
    existing = sys.modules.get("awomo")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
        raise ImportError("A foreign awomo package is already loaded; use a separate policy process")
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    module = importlib.import_module("awomo.awomo." + name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ImportError("Model components must come from this adapter")
    return module


def validate_contract(config):
    """Accept only the supported functional input contract; do not guess dimensions."""
    expected = {
        "action_dim": 80, "state_dim": 80, "action_horizon": 16, "action_scale": 1.0,
        "normalization": "zscore_absolute_joint", "gripper_source": "open_is_one",
        "gripper_model": "closed_is_one", "extra_padding48": "zero_before_transformer",
        "inactive80": "sigma_initial_noise_zero_clean", "image_layout": "hero_top",
        "video_size": [320, 288], "vl_image_size": [448, 448], "vl_text_layout": "prefix_v2",
        "history_slots": 20, "action_indices": list(ACTIVE),
        "condition_layers": [9, 18, 27], "summary_layers": [18, 27, 36],
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"Unsupported prepared checkpoint {name}: {config.get(name)!r}")
    if config.get("image_color_order") not in {"rgb", "bgr"}:
        raise ValueError("Unsupported checkpoint image_color_order")
    if float(config["scheduler"]["shift"]) != 5.0:
        raise ValueError("Prepared inference requires the declared shift=5")


def short_instruction_span(tokenizer, ids, instruction):
    for text in (" " + instruction.strip(), instruction.strip()):
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not 0 < len(tokens) < 3:
            continue
        starts = [i for i in range(len(ids) - len(tokens) + 1) if ids[i:i + len(tokens)] == tokens]
        if len(starts) == 1:
            return starts[0], starts[0] + len(tokens)
    return None


def inference_prompt_codec(encoder):
    base = bundled_module("datasets.robodojo.vlm_prompt").VLMPromptCodec

    class InferencePromptCodec(base):
        def _find_span(self, ids, instruction):
            result = super()._find_span(ids, instruction)
            if result is not None and result[1] - result[0] < 3:
                return short_instruction_span(self.tok, ids, instruction)
            return result if result is not None else short_instruction_span(self.tok, ids, instruction)

    return InferencePromptCodec(encoder.processor, encoder.image_size,
        history_image_size=encoder.history_image_size, history_pool=encoder.history_pool,
        history_period_s=encoder.history_period_s, max_history=encoder.history_num_slots)


@torch.no_grad()
def denoise_actions(model, encoded, *, action_indices=ACTIVE, action_horizon=16,
                    num_inference_steps=10, shift=5., seed=0, rand_device="cpu", scheduler_cls=None):
    if tuple(action_indices) != ACTIVE or action_horizon != 16 or num_inference_steps <= 0:
        raise ValueError("Invalid action/noise contract")
    if model.training:
        raise ValueError("Inference requires an eval-mode model")
    if scheduler_cls is None:
        scheduler_cls = bundled_module("models.backbones.schedulers.scheduler_continuous").FlowMatchScheduler
    reference = encoded["reference"]
    device, dtype = reference.device, reference.dtype
    batch = reference.shape[0]
    scheduler = scheduler_cls(num_inference_steps=num_inference_steps, num_train_timesteps=1000, shift=shift)
    seeds = [int(seed)] * batch if isinstance(seed, int) else list(seed)
    if len(seeds) != batch:
        raise ValueError("One seed is required per environment")
    images, actions = [], []
    for row_seed in seeds:
        generator = torch.Generator(device=rand_device).manual_seed(int(row_seed))
        images.append(torch.randn((1, *encoded["target"].shape[1:]), generator=generator,
                                  device=rand_device, dtype=dtype))
        actions.append(torch.randn((1, action_horizon, 80), generator=generator,
                                   device=rand_device, dtype=dtype))
    noisy_image = torch.cat(images).to(device)
    initial_action_noise = torch.cat(actions).to(device)
    noisy_action = initial_action_noise.clone()
    inactive = torch.ones(80, dtype=torch.bool, device=device)
    inactive[list(action_indices)] = False
    clean_zero = torch.zeros_like(noisy_action)
    for index, timestep in enumerate(scheduler.timesteps):
        expanded = timestep.expand(batch).to(device=device, dtype=torch.float32)
        known_path = scheduler.add_noise(clean_zero, initial_action_noise, expanded)
        noisy_action[..., inactive] = known_path[..., inactive]
        image_prediction, action_prediction = model(
            reference=reference, noisy_image=noisy_image, noisy_action=noisy_action,
            context=encoded["context"], text_mask=encoded["text_mask"], state=encoded["state"],
            timestep=expanded, reference_ids=encoded["reference_ids"],
            target_ids=encoded["target_ids"], condition_mask=encoded["condition_mask"],
        )
        final = index == num_inference_steps - 1
        noisy_image = scheduler.step(image_prediction, timestep, noisy_image, to_final=final)
        noisy_action = scheduler.step(action_prediction, timestep, noisy_action, to_final=final)
    noisy_action[..., inactive] = 0
    if not torch.isfinite(noisy_action).all() or not torch.isfinite(noisy_image).all():
        raise FloatingPointError("Non-finite denoising output")
    return {"action": noisy_action, "future_latents": noisy_image}


class InferenceRuntime:
    """Shared Awomo components with the RoboDojo condition and denoising API."""

    def __init__(self, checkpoint, *, qwen_path, vae_path, device="cuda", dtype=torch.bfloat16):
        from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel

        checkpoint = Path(checkpoint)
        loader = bundled_module("factory")
        components = bundled_module("model")
        encoder_module = bundled_module("model_vl")
        self.config = loader.read_checkpoint_config(checkpoint)
        validate_contract(self.config)
        self.device, self.torch_dtype = torch.device(device), dtype
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        payload = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        loader.validate_checkpoint_payload(payload)
        with torch.device("meta"):
            transformer = Flux2Transformer2DModel.from_config(payload["dit_config"])
            self.model = components.ActionPatchModel(transformer, 80, 80, 1.0)
        self.model.to_empty(device="cpu")
        del payload
        vlm = self.config
        self.vl_encoder = encoder_module.VLPromptEncoder(
            str(qwen_path), device=str(self.device), torch_dtype=torch.bfloat16,
            qwen_layers=vlm["condition_layers"], summary_layers=vlm["summary_layers"], max_prompt_len=768,
            lora={"enabled": True, "r": vlm["vl_lora"]["rank"], "alpha": vlm["vl_lora"]["alpha"]},
            image_size=vlm["vl_image_size"],
            history_image_size=vlm["vl_image_size"], history_num_slots=vlm["history_slots"],
            history_pool=(4, 4), history_period_s=1., history_fps=25.,
        )
        self.load_evidence = loader.load_checkpoint(self.model, self.vl_encoder, checkpoint,
                                                           verify_checksums=False)
        if self.load_evidence["model_keys"] != 171 or self.load_evidence["lora_keys"] != 504:
            raise ValueError("Expected all 171 model and 504 adapter tensors")
        self.model.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        self.vae = components.VideoVAE(AutoencoderKLFlux2.from_pretrained(str(vae_path), torch_dtype=torch.bfloat16))
        self.vae.to(self.device).eval().requires_grad_(False)
        self.vl_encoder.eval().requires_grad_(False)
        self.vl_encoder._codec = inference_prompt_codec(self.vl_encoder)
        self.video_size = tuple(vlm["video_size"])
        self.image_color_order = vlm["image_color_order"]
        self.image_layout = vlm["image_layout"]
        self.action_indices = ACTIVE
        self.scheduler_cls = bundled_module("models.backbones.schedulers.scheduler_continuous").FlowMatchScheduler
        LOG.info("PREPARED_CHECKPOINT_LOADED %s", json.dumps(self.load_evidence))

    @torch.no_grad()
    def encode_observation_batch(self, views, state, history_frames, history_valid_mask,
                                 instructions, vl_images, vl_history):
        batch = len(instructions)
        height, width = self.video_size
        if views.shape != (batch, 1, 3, height, width) or state.shape != (batch, 80):
            raise ValueError("Invalid current-image/state batch shape")
        if history_frames.shape != (batch, 20, 3, height, width) or history_valid_mask.shape != (batch, 20):
            raise ValueError("Invalid history batch shape")
        if history_valid_mask.dtype != torch.bool:
            raise ValueError("History validity must be boolean")
        views = views.to(device=self.device, dtype=torch.bfloat16)
        current = self.vae.encode(views[:, 0].unsqueeze(2))[:, :, 0].flatten(2).transpose(1, 2)
        valid = history_valid_mask.to(self.device)
        history = history_frames.to(device=self.device, dtype=torch.bfloat16)
        pooled = history.new_zeros(batch * 20, 16, 128)
        selected = history.reshape(batch * 20, 3, height, width)[valid.reshape(-1)]
        encoded_history = []
        for start in range(0, len(selected), 16):
            latent = self.vae.encode(selected[start:start + 16].unsqueeze(2))[:, :, 0]
            encoded_history.append(F.adaptive_avg_pool2d(latent.float(), (4, 4)).to(latent.dtype).flatten(2).transpose(1, 2))
        if encoded_history:
            pooled[valid.reshape(-1)] = torch.cat(encoded_history)
        history_ids = torch.cat([self.model.image_ids(4, 4, slot - 20, self.device) for slot in range(20)])
        current_ids = self.model.image_ids(height // 16, width // 16, 10, self.device)
        target_ids = self.model.image_ids(height // 16, width // 16, 0, self.device)
        mask = torch.cat([valid.repeat_interleave(16, dim=1),
                          torch.ones(batch, current.shape[1], dtype=torch.bool, device=self.device)], 1)
        vb = self.vl_encoder.build_infer_batch(instructions, vl_images, vl_history)
        context, text_mask = self.vl_encoder.infer_vlm(vb)
        return dict(reference=torch.cat([pooled.reshape(batch, 320, 128), current], 1).to(self.torch_dtype),
                    target=torch.zeros_like(current).to(self.torch_dtype), context=context.to(self.torch_dtype),
                    text_mask=text_mask, state=state.to(device=self.device, dtype=self.torch_dtype),
                    reference_ids=torch.cat([history_ids, current_ids]), target_ids=target_ids, condition_mask=mask)

    @torch.no_grad()
    def infer_action_flux2_batch(self, requests, *, dit_serial=True):
        if not requests:
            return []
        first = requests[0]
        for request in requests:
            if request["action_horizon"] != 16 or request["num_inference_steps"] != first["num_inference_steps"]:
                raise ValueError("Incompatible inference requests")
            if request["rand_device"] != first["rand_device"] or request["sigma_shift"] != 5.:
                raise ValueError("Incompatible random device or shift")
        encoded = self.encode_observation_batch(
            torch.cat([r["input_image"] for r in requests]), torch.cat([r["proprio"] for r in requests]),
            torch.cat([r["history_frames"] for r in requests]), torch.cat([r["history_valid_mask"] for r in requests]),
            [r["prompt"] for r in requests], [r["vl_images"][0] for r in requests], [r["vl_history"][0] for r in requests])
        options = dict(action_indices=ACTIVE, action_horizon=16, num_inference_steps=first["num_inference_steps"],
                       shift=5., rand_device=first["rand_device"], scheduler_cls=self.scheduler_cls)
        if dit_serial:
            outputs = []
            for i, request in enumerate(requests):
                row = {k: v if k in {"reference_ids", "target_ids"} else v[i:i + 1] for k, v in encoded.items()}
                outputs.append(denoise_actions(self.model, row, seed=int(request["seed"]), **options))
            output = {key: torch.cat([row[key] for row in outputs]) for key in ("action", "future_latents")}
        else:
            output = denoise_actions(self.model, encoded, seed=[int(r["seed"]) for r in requests], **options)
        return [{"action": output["action"][i], "future_latents": output["future_latents"][i:i + 1]}
                for i in range(len(requests))]
