"""
Batched inference wrapper for X-WAM, used by the XPolicyLab eval adapter.

This module loads an X-WAM checkpoint exactly like ``policy_server.py`` and
exposes a single batched entry point, :meth:`XWAMPolicy.infer_batch`, that:

1. Accepts a batch of multi-view RGB frames + raw proprio vectors + prompts.
2. Normalizes inputs with the dataset quantile statistics.
3. Runs ``model.generate`` once on the whole batch (the underlying model is
   already batch-capable; see ``runners/xwam_runner.py``).
4. Denormalizes the predicted **delta** actions and proprios and returns them
   per batch element.

The conversion from delta actions to the absolute end-effector poses that the
XPolicyLab environment expects is intentionally left to the adapter
(``policy/X_WAM/model.py``), which holds the current robot state anchor.
"""

import os
import sys

# Make the X-WAM package root importable (runners/, modules/, ...).
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_XWAM_ROOT = os.path.dirname(_EVAL_DIR)
if _XWAM_ROOT not in sys.path:
    sys.path.insert(0, _XWAM_ROOT)

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from omegaconf import OmegaConf
import lightning as L

from runners.xwam_runner import XWAMRunner


def resize_and_center_crop_tensor(tensor, resized_shape, crop_ratio, depth=False):
    """[B, V, C, H, W] -> resized + center-cropped (crop_ratio) -> back to size.

    Vendored from ``evaluation/policy_server.py`` to avoid importing that module
    (which pulls in zmq/tyro at load time).
    """
    B, V, _, _, _ = tensor.shape
    tensor = tensor.flatten(0, 1)
    tensor = TF.resize(tensor, size=resized_shape, interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
    tensor = tensor.unflatten(0, (B, V))
    H, W = resized_shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2
    out = []
    for b in range(B):
        out_b = []
        for v in range(V):
            img = tensor[b, v]
            img_cropped = TF.crop(img, top, left, crop_h, crop_w)
            interp = TF.InterpolationMode.NEAREST_EXACT if depth else TF.InterpolationMode.BILINEAR
            img_cropped = TF.resize(img_cropped, [H, W], interpolation=interp, antialias=False)
            out_b.append(img_cropped)
        out.append(torch.stack(out_b, dim=0))
    return torch.stack(out, dim=0)


def build_statistics(config):
    """Build quantile normalization arrays from ``config.dataset.statistics``.

    Vendored from ``evaluation/policy_server.py``. Supports single- and dual-arm
    configs; dims without explicit stats default to identity normalization.
    """
    stats = config.dataset.statistics
    has_right_arm = "proprio_right_ee_xyz" in stats.q01

    # States [16]: [l_xyz(3), l_quat(4), l_grip(1), r_xyz(3), r_quat(4), r_grip(1)]
    state_q01 = list(stats.q01.proprio_left_ee_xyz) + [-1.0] * 4 + list(stats.q01.gripper_pos)
    state_q99 = list(stats.q99.proprio_left_ee_xyz) + [1.0] * 4 + list(stats.q99.gripper_pos)
    if has_right_arm:
        state_q01 += list(stats.q01.proprio_right_ee_xyz) + [-1.0] * 4 + list(stats.q01.gripper_pos)
        state_q99 += list(stats.q99.proprio_right_ee_xyz) + [1.0] * 4 + list(stats.q99.gripper_pos)
    else:
        state_q01 += [-1.0] * 8
        state_q99 += [1.0] * 8

    # Actions: single-arm [7] / dual-arm [14], each arm = [xyz(3), aa(3), grip(1)].
    action_q01 = list(stats.q01.action_left_ee_xyz) + list(stats.q01.action_left_ee_axisangle) + list(stats.q01.gripper_action)
    action_q99 = list(stats.q99.action_left_ee_xyz) + list(stats.q99.action_left_ee_axisangle) + list(stats.q99.gripper_action)
    if has_right_arm:
        action_q01 += list(stats.q01.action_right_ee_xyz) + list(stats.q01.action_right_ee_axisangle) + list(stats.q01.gripper_action)
        action_q99 += list(stats.q99.action_right_ee_xyz) + list(stats.q99.action_right_ee_axisangle) + list(stats.q99.gripper_action)

    return (
        np.array(state_q01, dtype=np.float32),
        np.array(state_q99, dtype=np.float32),
        np.array(action_q01, dtype=np.float32),
        np.array(action_q99, dtype=np.float32),
        has_right_arm,
    )


class XWAMPolicy:
    """Loads an X-WAM checkpoint and runs batched action inference."""

    def __init__(
        self,
        exp_path: str,
        steps: str = "last",
        wan_checkpoint_dir: str | None = None,
        denoise_steps: int = 50,
        action_denoise_steps: int = 10,
        device: str = "cuda",
        compile_model: bool = True,
    ) -> None:
        config = OmegaConf.load(os.path.join(exp_path, "config.yaml"))
        config.sample_steps = int(denoise_steps)
        config.use_decoupled_inference = int(action_denoise_steps) > 0
        config.action_denoise_steps = int(action_denoise_steps)
        config.action_num = config.dataset.frame_skip // config.dataset.action_skip

        if wan_checkpoint_dir is not None:
            config.wan_checkpoint_dir = wan_checkpoint_dir
        if config.get("wan_checkpoint_dir") is None:
            raise ValueError(
                "Wan2.2-TI2V-5B checkpoint directory must be specified in "
                "config.yaml or via wan_checkpoint_dir."
            )

        self.config = config
        self.device = device

        # Quantile normalization arrays + arm layout (single vs dual arm).
        (
            self.state_q01,
            self.state_q99,
            self.action_q01,
            self.action_q99,
            self.has_right_arm,
        ) = build_statistics(config)
        self.action_dim = len(self.action_q01)

        L.seed_everything(config.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = XWAMRunner(config=config).to(device).bfloat16()
        ckpt = torch.load(
            os.path.join(
                exp_path,
                f"checkpoints/{steps}.ckpt/checkpoint/mp_rank_00_model_states.pt",
            ),
            map_location="cpu",
        )
        model.load_state_dict(ckpt["module"])
        model.eval()
        if compile_model:
            model.model = torch.compile(model.model)
        self.model = model

        # Geometry of the model outputs (per arm: xyz3 + axisangle3 + gripper1).
        self.video_size = tuple(config.dataset.video_size)
        self.crop_ratio = float(config.dataset.get("crop_ratio", 0.95))

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    def _normalize_proprios(self, proprios: np.ndarray) -> np.ndarray:
        """Quantile-normalize a [B, 16] raw proprio batch into [-1, 1]."""
        proprio_norm = 2.0 * (proprios - self.state_q01) / (self.state_q99 - self.state_q01) - 1.0
        if not self.has_right_arm:
            proprio_norm[:, 8:] = 0.0
        return proprio_norm.astype(np.float32)

    def _prepare_video(self, video: np.ndarray) -> torch.Tensor:
        """video: [B, V, H, W, C] uint8 -> [B, V, C, H', W'] bf16 in [-1, 1]."""
        rgb = torch.from_numpy(video).to(self.device)
        rgb = rgb.float() / 127.5 - 1.0
        rgb = rgb.bfloat16()
        rgb = rearrange(rgb, "b v h w c -> b v c h w")
        rgb = resize_and_center_crop_tensor(rgb, self.video_size, self.crop_ratio)
        return rgb

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def infer_batch(
        self,
        video: np.ndarray,
        proprios: np.ndarray,
        prompts: list[str],
        seeds: list[int],
        cfg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one batched X-WAM forward pass.

        Args:
            video:    [B, V, H, W, C] uint8 RGB, view order = head/left/right.
            proprios: [B, 16] raw proprio vectors (un-normalized), layout
                      [l_xyz3, l_quat_wxyz4, l_grip1, r_xyz3, r_quat_wxyz4, r_grip1].
            prompts:  list[str] of length B (language instruction per element).
            seeds:    list[int] of length B for reproducible noise.
            cfg:      classifier-free guidance scale (0 disables CFG).

        Returns:
            actions:  [B, Ta, action_dim] denormalized **delta** actions.
                      Single-arm: [dxyz3, daxisangle3, dgrip1].
                      Dual-arm:   left(7) + right(7).
            proprios: [B, Tp, 16] denormalized predicted proprios.
        """
        proprios = np.asarray(proprios, dtype=np.float32)
        if proprios.ndim == 1:
            proprios = proprios[None, :]
        batch_size = proprios.shape[0]
        assert len(prompts) == batch_size, f"prompts len {len(prompts)} != batch {batch_size}"
        assert len(seeds) == batch_size, f"seeds len {len(seeds)} != batch {batch_size}"

        rgb = self._prepare_video(np.asarray(video))
        proprio_norm = self._normalize_proprios(proprios)
        proprio = torch.from_numpy(proprio_norm).bfloat16().to(self.device)

        _, xt_actions, xt_proprios, _ = self.model.generate(
            rgb,
            proprio,
            list(prompts),
            seeds=list(seeds),
            early_stop=True,
            cfg=float(cfg),
            run_depth=False,
        )

        actions = xt_actions.float().cpu().numpy()  # [B, Ta, >=action_dim]
        proprios_out = xt_proprios.float().cpu().numpy()  # [B, Tp, 16]

        # Denormalize proprios over the full 16-dim layout.
        proprios_out = (proprios_out + 1.0) / 2.0 * (self.state_q99 - self.state_q01) + self.state_q01

        # Denormalize actions; only the first action_dim columns are meaningful.
        actions = actions[:, :, : self.action_dim]
        actions = (actions + 1.0) / 2.0 * (self.action_q99 - self.action_q01) + self.action_q01
        if not self.has_right_arm:
            actions[:, :, 6] *= -1.0  # invert gripper for single-arm (robocasa)

        return actions, proprios_out


def get_model(usr_args: dict) -> XWAMPolicy:
    """Build an :class:`XWAMPolicy` from a flat XPolicyLab-style config dict."""
    def _opt(*keys, default=None):
        for k in keys:
            v = usr_args.get(k)
            if v is not None and not (isinstance(v, str) and v.strip().lower() in {"", "none", "null"}):
                return v
        return default

    exp_path = _opt("exp_path", "checkpoint_path")
    if exp_path is None:
        raise FileNotFoundError("X-WAM requires exp_path (experiment dir containing config.yaml).")

    return XWAMPolicy(
        exp_path=str(exp_path),
        steps=str(_opt("steps", default="last")),
        wan_checkpoint_dir=_opt("wan_checkpoint_dir"),
        denoise_steps=int(_opt("denoise_steps", "num_inference_steps", default=50)),
        action_denoise_steps=int(_opt("action_denoise_steps", default=10)),
        device=str(_opt("device", default="cuda")),
        compile_model=bool(_opt("compile_model", default=True)),
    )
