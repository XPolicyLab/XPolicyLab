"""把固定 LoRA 权重接到全量评测使用的原生推理实现。"""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

try:
    from .runtime import load_official
    from .lora_policy import DuoFlowPolicy
except ImportError:
    from runtime import load_official
    from lora_policy import DuoFlowPolicy


def load_model(adapter_path):
    adapter_path = Path(adapter_path).expanduser().resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    variant = "FT2T10k"
    if adapter_path.suffix == ".safetensors":
        state = {"lora": load_file(str(adapter_path), device="cpu"), "adapter": {}}
    else:
        # 原始检查点仅用于本地续训/对照复现；公开发布使用 safetensors。
        payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
        if payload.get("mode") != "FT" or not payload.get("run_complete") or payload.get("update") != 10000:
            raise ValueError("Expected the completed ordinary-FT or FT2T 10k checkpoint")
        if payload.get("framework_variant") not in (None, "FT", "FT2T"):
            raise ValueError("Only the ordinary-FT control and fixed FT2T branch are supported")
        state = payload["adapter_state"]
        variant = "FT2T10k" if payload.get("framework_variant") == "FT2T" else "FT10k"
    if state.get("adapter"):
        raise ValueError("This release does not use an action-head residual module")
    if sum(t.numel() for t in state["lora"].values()) != 4046848:
        raise ValueError("The released rank-8 LoRA must contain 4,046,848 parameters")
    model = load_official()
    if model.n_action_steps != 50 or model.num_steps != 10 or model.max_action_dim != 32:
        raise ValueError("Backbone configuration differs from the evaluated 50-step/10-NFE model")
    policy = DuoFlowPolicy(model, mode="FT")
    policy.configure_trainable(rank=8, alpha=16)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state["lora"]) != expected:
        raise ValueError("Adapter tensor names do not match the frozen backbone")
    policy.load_trainable_state_dict(state, strict=True)
    policy.to("cuda").eval()
    model.eval()
    # 与原评测一致：服务初始化后设一次 seed，不在每次查询时重置。
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model.server_info.update(
        method="DuoFlow" if variant == "FT2T10k" else "ordinary_FT", variant=variant, prediction_horizon=50,
        execution_horizon=30, denoise_steps=10, model_dtype="torch.bfloat16",
        adapter_dtype="float32", checkpoint=str(adapter_path),
        checkpoint_repo="open-gigaai/GigaBrain-0.7-3.5B-RoboTwin2.0-Clean",
        checkpoint_revision="3de7c7e60e5779d016c8f148eed19c4b99cd2a95",
        full_joint_observation_and_fusion=True, external_execution_controller=False,
        inference_rng_initial_seed=0, per_query_rng_reset=False,
    )
    return model


class DuoFlowInference:
    def __init__(self, adapter_path):
        self.model = load_model(adapter_path)

    @torch.inference_mode()
    def predict(self, state, images, instruction, *, reset):
        request = {
            "observation.state": torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)),
            "task": instruction,
            "reset_observation_memory": bool(reset),
        }
        for name, rgb in images.items():
            # 对已解码 uint8 RGB 仅执行原生 float_native 转换。
            value = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
            request[f"observation.images.{name}"] = torch.from_numpy(value)
        action = self.model.inference(request)
        return action.detach().float().cpu().numpy()
