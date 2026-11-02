"""Construct the historical policy without its training data or latent cache."""
import os
from pathlib import Path
import torch
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from .assets import MODEL_FINGERPRINT, manifest_fingerprint, tensor_manifest, validate_bundle
from .runtime import historical_module


def load_policy(model_cfg, policy_dir, action_dim):
    root = resolve_checkpoint_root(model_cfg, Path(policy_dir) / "checkpoints")
    if not root.is_dir():
        raise ValueError("Pass the exported inference bundle directory, not a raw training checkpoint")
    config, tasks = validate_bundle(root)
    if int(config["action_dim"]) != action_dim:
        raise ValueError("Robot action dimensions do not match the checkpoint")
    wan = model_cfg.get("wan_checkpoint_dir") or os.environ.get("WAN_CHECKPOINT_DIR")
    state = torch.load(root / "model.pt", map_location="cpu", mmap=True, weights_only=True)
    if manifest_fingerprint(tensor_manifest(state)) != MODEL_FINGERPRINT:
        raise ValueError("Model tensors differ from the retained s742500 checkpoint")
    if not wan:
        raise ValueError("Set WAN_CHECKPOINT_DIR to the official Wan2.2-TI2V-5B assets")
    wan = Path(wan).expanduser().resolve()
    if not (wan / "Wan2.2_VAE.pth").is_file():
        raise FileNotFoundError(wan / "Wan2.2_VAE.pth")
    shapes = {key: (3, 384, 320) for key in ("rgb_head", "rgb_head_future", "rgb_head_last")}
    shapes["robot_state"] = (action_dim,)
    encoder_cls = historical_module("psi_policy.model.wan.official_obs_encoder").WanOfficialObsEncoder
    # Historical inference explicitly used live VAE encoding. No cache lookup or dataset access.
    encoder = encoder_cls(obs_shapes=shapes, obs_horizon=1, target_horizon=16,
                          wan_checkpoint_dir=str(wan), image_key="rgb_head",
                          target_key="rgb_head_future", fallback_target_key="rgb_head_last", vae_dtype="bf16")
    options = dict(config["policy"])
    target = options.pop("_target_")
    expected = "hyp_bslii.policy.wan_bslii_v58_light_lang_policy.WanBSLIIv58LightLangPolicy"
    if target != expected:
        raise ValueError("Unsupported historical policy class")
    options.update(wan_checkpoint_dir=str(wan), task_ctx_path=str(root / "task_context.pt"),
                   obs_encoder=encoder, action_dim=action_dim, action_horizon=16)
    policy_cls = historical_module(target.rpartition(".")[0]).WanBSLIIv58LightLangPolicy
    policy = policy_cls(**options)
    policy._apply_dit_train_mode("full_train", {})
    result = policy.load_state_dict(state, strict=True)
    policy.num_inference_steps = int(model_cfg.get("num_inference_steps", 30))
    if policy.num_inference_steps != 30:
        raise ValueError("The historical evaluation profile requires 30 denoising steps")
    device = str(model_cfg.get("device") or "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real video/action inference")
    policy.to(device).eval()
    policy.requires_grad_(False)
    print(f"[RAP_WAM] strict checkpoint load: {len(state)} tensors; {result}; device={device}", flush=True)
    return policy, tasks, root, wan
