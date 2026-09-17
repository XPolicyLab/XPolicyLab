"""Load a MoPA checkpoint and predict joint-action chunks from RGB observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .common import FORMAT_VERSION, denormalize, joint_fields, normalize, rgb_image, validate_statistics
from .models.policy import ArmQueryPolicy


class Policy:
    """Standalone inference on unnormalized joint states and decoded RGB images."""

    def __init__(self, checkpoint, *, device="cpu", base_vlm=None, dtype=None):
        checkpoint = Path(checkpoint).expanduser()
        directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
        weights = directory / "model.pt" if checkpoint.is_dir() else checkpoint
        self.config = json.loads((directory / "config.json").read_text())
        if self.config.get("format_version") != FORMAT_VERSION:
            raise ValueError("Expected a MoPA manipulation-only checkpoint with one bank of eight queries")
        self.action_dim = int(self.config["action_dim"])
        self.state_dim = int(self.config["state_dim"])
        dim = sum(size for _, size in joint_fields(self.config["robot_action_dim_info"]))
        if self.state_dim != dim or self.action_dim != dim:
            raise ValueError("Checkpoint dimensions disagree with its joint layout")
        self.cameras = self.config["cameras"]
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("Checkpoint cameras must be nonempty and unique")
        self.image_size = self.config["image_size"]
        if len(self.image_size) != 2 or any(int(size) <= 0 for size in self.image_size):
            raise ValueError("Checkpoint image_size must contain positive height and width")
        self.statistics = json.loads((directory / "dataset_statistics.json").read_text())
        for kind, size in (("state", self.state_dim), ("action", self.action_dim)):
            validate_statistics(self.statistics[kind], size)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if base_vlm is not None:
            self.config["base_vlm"] = str(base_vlm)
        self.config["device"] = str(self.device)
        self.config["backbone_dtype"] = "float32" if self.device.type == "cpu" else (dtype or self.config.get("backbone_dtype", "bfloat16"))
        self.model = ArmQueryPolicy(self.config)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval()

    def predict(self, images, instructions, states):
        """Return physical joint actions [batch, 4, dim], with cameras in saved order."""
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim or not np.isfinite(states).all():
            raise ValueError(f"Expected finite joint states [batch, {self.state_dim}]")
        batch = states.shape[0]
        if batch == 0 or len(images) != batch or len(instructions) != batch:
            raise ValueError("Images, instructions and states need matching nonempty batches")
        if any(not isinstance(text, str) or not text.strip() for text in instructions):
            raise ValueError("Every observation needs a nonempty instruction string")
        if any(len(views) != len(self.cameras) for views in images):
            raise ValueError("Every observation must supply all cameras in checkpoint order")
        images = [[rgb_image(frame, self.image_size) for frame in views] for views in images]
        state = torch.as_tensor(normalize(states, self.statistics["state"]), device=self.device)
        with torch.inference_mode():
            actions = self.model.predict_action(images, list(instructions), state).float().cpu().numpy()
        expected = (batch, int(self.config["action_horizon"]), self.action_dim)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions of shape {expected}, got {actions.shape}")
        return denormalize(actions, self.statistics["action"])
