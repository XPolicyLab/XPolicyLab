import os
import sys
import numpy as np
import cv2
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.join(CURRENT_DIR, "RoboOrchardLab")

# XPolicyLab camera key → HoloBrain camera name (matches process_data.py CAMERA_MAP inverse)
_XPOLICY_TO_HOLOBRAIN_CAM = {
    "cam_head": "front_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


class Model(ModelTemplate):

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_chunk_size = int(model_cfg.get("action_chunk_size", 16))

        if UPSTREAM_DIR not in sys.path:
            sys.path.insert(0, UPSTREAM_DIR)

        from robo_orchard_lab.models.holobrain.pipeline import HoloBrainInferencePipeline

        model_dir = model_cfg.get("model_dir") or model_cfg.get("checkpoint_path")
        if not model_dir:
            raise ValueError(
                "[HoloBrain] deploy.yml must set 'model_dir' to the exported pipeline "
                "directory produced by:\n"
                "  cd XPolicyLab/policy/HoloBrain/RoboOrchardLab/projects/holobrain\n"
                "  python3 scripts/export.py --config <config> --workspace <workspace>\n"
            )

        inference_prefix = model_cfg.get("inference_prefix", "inference")
        gpu_id = model_cfg.get("gpu_id")
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"

        self.pipeline = HoloBrainInferencePipeline.load_pipeline(
            directory=model_dir,
            inference_prefix=inference_prefix,
            device=device,
            load_weights=True,
            load_impl="native",
        )
        self.pipeline.model.eval()
        self._last_model_input = None
        print(f"[HoloBrain] Pipeline loaded from {model_dir} on {device}")

    def _build_model_input(self, obs):
        from robo_orchard_lab.models.holobrain.processor import MultiArmManipulationInput

        images = {}
        depths = {}
        intrinsics = {}
        t_world2cam = {}

        for xp_cam, hb_cam in _XPOLICY_TO_HOLOBRAIN_CAM.items():
            cam_data = obs.get("vision", {}).get(xp_cam)
            if cam_data is None:
                continue

            # --- image: XPolicyLab is RGB; resize to (240, 320, 3) ---
            img = np.asarray(cam_data["color"], dtype=np.uint8)
            if img.shape[:2] != (240, 320):
                img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
            assert img.shape == (240, 320, 3), f"[HoloBrain] Unexpected image shape {img.shape}"
            images[hb_cam] = [img]

            # --- depth: single-channel float64 in metres (240, 320) ---
            raw_depth = cam_data.get("depth")
            if raw_depth is not None:
                d = np.asarray(raw_depth, dtype=np.float32)
                if d.ndim == 3:
                    d = d[..., 0]
                d = cv2.resize(d, (320, 240), interpolation=cv2.INTER_NEAREST).astype(np.float64)
                d = d / 1000.0  # mm → m
            else:
                d = np.zeros((240, 320), dtype=np.float64)
            depths[hb_cam] = [d]

            # --- intrinsic: scale 3×3 → 4×4 for target 320×240 ---
            src_shape = cam_data.get("shape", (480, 640))
            src_h, src_w = int(src_shape[0]), int(src_shape[1])
            raw_k = np.asarray(
                cam_data.get("intrinsic_matrix", np.eye(3)), dtype=np.float64
            )
            k4 = np.eye(4, dtype=np.float64)
            if raw_k.shape == (3, 3):
                k4[:3, :3] = raw_k
                k4[0, :] *= 320.0 / float(src_w)
                k4[1, :] *= 240.0 / float(src_h)
                k4[2, 2] = 1.0
            elif raw_k.shape == (4, 4):
                k4 = raw_k.copy()
            intrinsics[hb_cam] = k4

            # --- extrinsic: use whatever the obs provides as world2cam ---
            ext_raw = cam_data.get("extrinsics_matrix") or cam_data.get("extrinsic_matrix")
            if ext_raw is not None:
                ext = np.asarray(ext_raw, dtype=np.float64)
                if ext.shape in ((3, 4), (4, 4)):
                    pad = np.eye(4, dtype=np.float64)
                    pad[: ext.shape[0], : ext.shape[1]] = ext
                    t_world2cam[hb_cam] = pad

        # --- joint state: pack to flat vector [arm0, ee0, arm1, ee1, ...] ---
        joint_state = pack_robot_state(
            obs, self.action_type, self.robot_action_dim_info, source_type="obs"
        ).astype(np.float32)

        instruction = obs.get("instruction") or ""

        return MultiArmManipulationInput(
            image=images,
            depth=depths,
            intrinsic=intrinsics,
            t_world2cam=t_world2cam if t_world2cam else None,
            history_joint_state=[joint_state],
            instruction=instruction,
        )

    def update_obs(self, obs):
        self._last_model_input = self._build_model_input(obs)

    def update_obs_batch(self, obs_list):
        self._last_model_input = [self._build_model_input(obs) for obs in obs_list]

    def get_action(self):
        if self._last_model_input is None:
            raise RuntimeError("[HoloBrain] Call update_obs before get_action.")
        with torch.no_grad():
            output = self.pipeline(self._last_model_input)
        # output.action shape: (chunk_size, total_action_dim)
        action_chunk = output.action.cpu().numpy()
        use_len = min(len(action_chunk), self.action_chunk_size)
        return [
            unpack_robot_state(
                action_chunk[i],
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
            for i in range(use_len)
        ]

    def get_action_batch(self, env_idx_list):
        if self._last_model_input is None:
            raise RuntimeError("[HoloBrain] Call update_obs_batch before get_action_batch.")
        model_inputs = self._last_model_input
        if not isinstance(model_inputs, list):
            model_inputs = [model_inputs] * len(env_idx_list)
        result = []
        for i, _ in enumerate(env_idx_list):
            inp = model_inputs[i] if i < len(model_inputs) else model_inputs[-1]
            with torch.no_grad():
                output = self.pipeline(inp)
            action_chunk = output.action.cpu().numpy()
            use_len = min(len(action_chunk), self.action_chunk_size)
            result.append(
                [
                    unpack_robot_state(
                        action_chunk[j],
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                    for j in range(use_len)
                ]
            )
        return result

    def reset(self):
        self._last_model_input = None
