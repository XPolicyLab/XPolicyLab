"""Strict inference restoration extracted from upstream wam.training.checkpoint.

No optimizer, scheduler state, training resume or initialization migrations.
"""
import copy
import json
import torch
from wam.model.modules.conditioning.frame_attention import JOINT96_ACTION_VIDEO_ATTENTION


def joint96_checkpoint_metadata(model):
    if not bool(getattr(model, "joint96_three_view", False)):
        return None
    paths = copy.deepcopy(getattr(model, "model_paths", {}))
    encoder = paths.get("joint_visual_encoder", {})
    if not encoder.get("source_sha256") or not encoder.get("stvae"):
        raise ValueError("Joint96 checkpoint requires frozen encoder source and checkpoint metadata")
    metadata = {"schema": "internw0.local_joint96.weights.v1", "model_paths": paths,
                "channel_order": ["wan_vae48", "stvae48"], "camera_order": ["main", "left", "right"],
                "action_dim": 80, "proprio_dim": 80,
                "action_video_attention": JOINT96_ACTION_VIDEO_ATTENTION}
    prompt = getattr(getattr(model, "action_expert", None), "embodiment_prompt", None)
    if prompt is not None:
        metadata["soft_prompt"] = prompt.metadata()
        contract = getattr(model, "soft_prompt_training_contract", None)
        if contract is not None:
            metadata["soft_prompt"]["training_contract"] = copy.deepcopy(contract)
    # Metadata must not accidentally carry full frozen encoder tensors.
    json.dumps(metadata, allow_nan=False)
    return metadata


def _joint96_identity(metadata):
    if not isinstance(metadata, dict) or metadata.get("schema") != "internw0.local_joint96.weights.v1":
        raise ValueError("Missing or incompatible Joint96 checkpoint metadata")
    attention = metadata.get("action_video_attention")
    if not isinstance(attention, str) or not attention:
        raise ValueError(
            "Joint96 checkpoint identity missing a valid action_video_attention; "
            "legacy attention semantics must be verified before migration"
        )
    encoder = metadata["model_paths"]["joint_visual_encoder"]
    stvae = encoder["stvae"]
    identity = {
        key: metadata[key]
        for key in (
            "channel_order", "camera_order", "action_dim", "proprio_dim",
            "action_video_attention",
        )
    }
    for key in ("source_sha256", "posterior", "per_camera_input_hw", "temporal_compression", "extra_latent_scaling"):
        identity[key] = encoder.get(key)
    if encoder.get("source_root") == "bundled://dino-stvae-v1":
        from wam.model.modules.codecs.dino_stvae.provenance import canonical_source_identity
        identity["source_sha256"] = canonical_source_identity(encoder)
    if "current_conditioning" in encoder:
        if encoder["current_conditioning"] != "independent_single_frame_v1":
            raise ValueError("Unknown Joint96 current_conditioning")
        identity["current_conditioning"] = encoder["current_conditioning"]
    for key in ("stvae_checkpoint_sha256", "teacher_sha256", "stvae_step", "teacher_checkpoint_state",
                "teacher_output_source", "teacher_preprocessing", "dinov3_source_commit"):
        identity[key] = stvae.get(key)
    for key in ("source_sha256", "stvae_checkpoint_sha256", "teacher_sha256", "stvae_step"):
        if not identity[key]:
            raise ValueError(f"Joint96 checkpoint identity missing {key}")
    if "soft_prompt" in metadata:
        from wam.embodiment_registry import EmbodimentRegistry
        prompt = metadata["soft_prompt"]
        registry = prompt.get("registry", {})
        EmbodimentRegistry.validate(registry)
        if (prompt.get("schema") != "wam.action_soft_prompt.v1"
                or EmbodimentRegistry.content_hash(registry) != prompt.get("registry_sha256")
                or prompt.get("num_embodiments") != registry["num_embodiments"]
                or any(type(prompt.get(key)) is not int or prompt[key] <= 0 for key in ("num_tokens", "context_dim"))):
            raise ValueError("Invalid soft prompt checkpoint identity")
        identity["soft_prompt"] = copy.deepcopy(prompt)
        # Provenance stays in the payload and full-state resume contract, but
        # inference does not instantiate the training dataset/package.
        identity["soft_prompt"].pop("training_contract", None)
    return identity


def _load_joint96_checkpoint(model, payload):
    """Restore an inference policy only after validating its identity and all tensors."""
    if _joint96_identity(payload.get("joint96_metadata")) != _joint96_identity(joint96_checkpoint_metadata(model)):
        raise ValueError("Joint96 encoder/channel/action/camera identity mismatch")
    modules = (("mot", model.mot), ("proprio_encoder", model.proprio_encoder),
               ("action_proprio_encoder", model.action_proprio_encoder))
    for name, module in modules:
        incoming = payload.get(name)
        if module is None:
            if incoming is not None:
                raise ValueError(f"Joint96 unexpected module: {name}")
            continue
        expected = module.state_dict()
        if not isinstance(incoming, dict) or set(incoming) != set(expected):
            raise ValueError(f"Joint96 strict state keys mismatch: {name}")
        for key, tensor in incoming.items():
            if not torch.is_tensor(tensor) or tensor.shape != expected[key].shape:
                raise ValueError(f"Joint96 strict tensor shape mismatch: {name}.{key}")
    for name, module in modules:
        if module is not None:
            module.load_state_dict(payload[name], strict=True)
    return payload
