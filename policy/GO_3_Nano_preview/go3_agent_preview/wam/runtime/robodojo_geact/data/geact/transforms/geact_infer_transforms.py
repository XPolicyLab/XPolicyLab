from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _caption_from_obs(data: dict[str, Any]) -> str:
    if "caption" in data:
        prompt = data["caption"]
    elif "prompt" in data:
        prompt = data["prompt"]
    else:
        raise KeyError("prompt or caption required")
    if isinstance(prompt, str):
        return prompt
    return prompt[0]


@dataclass
class GeactCorobotPrepare:
    cam_order: Sequence[str]
    target_h: int
    target_w: int
    s_mean: torch.Tensor
    s_std: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    default_num_ae_steps: int
    default_dense_action_steps: int
    legacy_state_layout: str = "identity"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = dict(data)
        if "video" in out and "state" in out and torch.is_tensor(out["video"]):
            out["caption"] = _caption_from_obs(out)
            if "num_ae_steps" not in out:
                out["num_ae_steps"] = self.default_num_ae_steps
            if "dense_action_steps" not in out:
                out["dense_action_steps"] = self.default_dense_action_steps
            return out

        if "obs" in out:
            video, state = self._from_legacy(out)
        else:
            raise KeyError(
                "GeactCorobotPrepare expects obs/state "
                f"or prebuilt video/state; got keys={sorted(out)}"
            )

        out["video"] = video
        out["state"] = state
        out["caption"] = _caption_from_obs(out)
        if "num_ae_steps" not in out:
            out["num_ae_steps"] = self.default_num_ae_steps
        if "dense_action_steps" not in out:
            out["dense_action_steps"] = self.default_dense_action_steps
        return out

    def _from_legacy(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        imgs = np.array(obs["obs"]) if isinstance(obs["obs"], list) else obs["obs"]
        if imgs.ndim == 5:
            video = torch.from_numpy(imgs.astype(np.float32)).permute(0, 4, 1, 2, 3)
        elif imgs.ndim == 4:
            video = (
                torch.from_numpy(imgs.astype(np.float32))
                .permute(3, 0, 1, 2)
                .unsqueeze(0)
            )
        else:
            raise ValueError(f"Invalid video shape: {imgs.shape}")
        if video.max() > 1.0:
            video = (video / 127.5) - 1.0

        state = torch.from_numpy(np.array(obs["state"], dtype=np.float32, copy=True)).to(
            self.device, dtype=torch.float32
        )
        state = self._adapt_legacy_state(state)
        state = (state - self.s_mean) / self.s_std
        if state.ndim == 1:
            state = state.view(1, 1, -1)
        elif state.ndim == 2:
            state = state.unsqueeze(1)
        return (
            video.to(self.device, dtype=self.dtype),
            state.to(self.device, dtype=self.dtype),
        )

    def _adapt_legacy_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.legacy_state_layout == "identity":
            return state
        if self.legacy_state_layout != "robodojo_x5":
            raise ValueError(
                "unsupported legacy_state_layout: "
                f"{self.legacy_state_layout!r}"
            )
        if state.shape[-1] != 14:
            raise ValueError(
                "robodojo_x5 expects [left6, left_gripper, right6, "
                f"right_gripper] (14D), got {tuple(state.shape)}"
            )
        if self.s_mean.numel() != 24:
            raise ValueError(
                "robodojo_x5 requires 24D g2_sim statistics, got "
                f"{self.s_mean.numel()}D"
            )

        target = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15)
        expanded = self.s_mean.to(device=state.device, dtype=state.dtype).expand(
            *state.shape[:-1], -1
        ).clone()
        expanded[..., list(target)] = state
        return expanded


@dataclass
class GeactCorobotFormat:
    robot_name: str
    action_dim: int
    default_dense_action_steps: int
    return_head_waist: bool = False

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in data:
            raise KeyError("GeactCorobotFormat requires 'actions' from policy.step")
        actions = data["actions"]
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if "dense_action_steps" in data and data["dense_action_steps"] is not None:
            dense_steps = data["dense_action_steps"]
        else:
            dense_steps = self.default_dense_action_steps
        arr = arr[:dense_steps, : self.action_dim]
        if self.robot_name in {"sl_acone", "sl_arxvr", "sl_agilex", "robodojo", "robodojo_real"}:
            if self.action_dim != 14:
                raise ValueError(
                    f"{self.robot_name} requires a native 14D action output, got {self.action_dim}"
                )
            result: dict[str, Any] = {
                "left_arm": {"kind": "JOINT_ABS", "values": arr[:, 0:6].tolist()},
                "right_arm": {"kind": "JOINT_ABS", "values": arr[:, 7:13].tolist()},
                "left_effector": arr[:, 6:7].tolist(),
                "right_effector": arr[:, 13:14].tolist(),
            }
            return {"result": result, "actions": actions, "dense_action_steps": dense_steps}

        result: dict[str, Any] = {
            "left_arm": {"kind": "JOINT_ABS", "values": arr[:, 0:7].tolist()},
            "right_arm": {"kind": "JOINT_ABS", "values": arr[:, 8:15].tolist()},
            "left_effector": arr[:, 7:8].tolist(),
            "right_effector": arr[:, 15:16].tolist(),
        }
        if (
            self.return_head_waist
            and self.robot_name in ("g2_sim", "g2_real", "g2_90d")
            and self.action_dim >= 24
        ):
            result["head"] = arr[:, 16:19].tolist()
            result["waist"] = {
                "kind": "JOINT_ABS",
                "values": arr[:, 19:24].tolist(),
            }
        return {
            "result": result,
            "actions": actions,
            "dense_action_steps": dense_steps,
        }
