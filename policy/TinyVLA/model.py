import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


TINYVLA_DIR = Path(__file__).resolve().parent / "tinyvla"
if str(TINYVLA_DIR) not in sys.path:
    sys.path.append(str(TINYVLA_DIR))

from eval_real_franka import llava_pythia_act_policy
from llava_pythia.conversation import conv_templates
from llava_pythia.mm_utils import tokenizer_image_token
from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.action_type = model_cfg["action_type"]
        self.camera_keys = list(model_cfg["camera_keys"])
        self.conv_mode = model_cfg["conv_mode"]
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])

        self.policy = llava_pythia_act_policy({
            "model_path": model_cfg["model_path"],
            "model_base": model_cfg["model_base"],
            "enable_lora": model_cfg["enable_lora"],
            "conv_mode": model_cfg["conv_mode"],
        })
        self.policy.policy.eval()

        with open(model_cfg["stats_path"], "rb") as f:
            self.stats = pickle.load(f)

        self.latest_obs = {}
        self.latest_env_idx_list = []

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self.latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs["env_idx"])
            self.latest_obs[env_idx] = self._encode_obs(obs)
            self.latest_env_idx_list.append(env_idx)

    def get_action(self):
        return self.get_action_batch([self.latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list):
        action_dict_list = []
        for env_idx in env_idx_list:
            env_idx = int(env_idx)
            curr_image, robot_state, raw_lang = self.latest_obs[env_idx]

            batch = self._build_policy_batch(curr_image, robot_state, raw_lang)
            with torch.inference_mode():
                all_actions = self.policy.policy(**batch, eval=True)

            action_chunk = all_actions[0].detach().cpu().to(torch.float32).numpy()
            action_chunk = self._post_process_action(action_chunk)

            action_dict_list.append(
                unpack_robot_state(
                    action_chunk,
                    action_type=self.action_type,
                    robot_action_dim_info=self.robot_action_dim_info,
                    source_type="obs",
                )
            )
        return action_dict_list

    def reset(self):
        self.latest_obs.clear()
        self.latest_env_idx_list.clear()

    def _encode_obs(self, obs):
        cam_chws = []
        for cam_key in self.camera_keys:
            camera_obs = obs["vision"][cam_key]
            raw = camera_obs["color"] if "color" in camera_obs else camera_obs["colors"]
            if isinstance(raw, (bytes, bytearray, np.bytes_)):
                rgb = decode_image_bit(raw)
            else:
                raw = np.asarray(raw)
                if raw.ndim == 1:
                    rgb = decode_image_bit(raw)
                else:
                    rgb = raw
            rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
            cam_chws.append(np.transpose(rgb, (2, 0, 1)))

        stacked = np.stack(cam_chws, axis=0).astype(np.float32) / 255.0
        curr_image = torch.from_numpy(stacked).cuda().unsqueeze(0)

        state_vec = pack_robot_state(
            obs=obs,
            action_type=self.action_type,
            robot_action_dim_info=self.robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)
        state_vec = (state_vec - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        robot_state = torch.from_numpy(state_vec).cuda().unsqueeze(0)

        raw = obs["instruction"] if "instruction" in obs else obs["instructions"]
        if isinstance(raw, (list, tuple, np.ndarray)):
            raw = raw[0]
        if isinstance(raw, (bytes, bytearray, np.bytes_)):
            raw = raw.decode("utf-8")
        raw_lang = str(raw)

        return curr_image, robot_state, raw_lang

    def _build_policy_batch(self, curr_image, robot_state, raw_lang):
        """Same as llava_pythia_act_policy.process_batch_to_llava, but supports the
        3-camera (left wrist / right wrist / head) token_cat layout via images_top."""
        policy = self.policy

        if curr_image.dim() == 5:  # (1, num_cam, C, H, W)
            curr_image = curr_image.squeeze(0)

        image_tensors = []
        for cam_image in torch.chunk(curr_image, curr_image.shape[0], dim=0):
            cam_image = policy.expand2square(
                cam_image, tuple(x for x in policy.image_processor.image_mean)
            )
            pixel_values = policy.image_processor.preprocess(
                cam_image,
                return_tensors="pt",
                do_normalize=True,
                do_rescale=False,
                do_center_crop=False,
            )["pixel_values"]
            image_tensors.append(
                pixel_values.to(policy.policy.device, dtype=policy.policy.dtype)
            )

        if policy.policy.config.mm_use_im_start_end:
            inp = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                + "\n" + raw_lang
            )
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + raw_lang
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " <|endoftext|>"

        input_ids = (
            tokenizer_image_token(
                prompt, policy.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(policy.policy.device)
        )
        attn_mask = input_ids.ne(policy.tokenizer.pad_token_id)

        batch = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensors[0],
            images_r=image_tensors[1],
            states=robot_state.to(policy.policy.device, dtype=policy.policy.dtype),
        )
        if len(image_tensors) == 3:
            batch["images_top"] = image_tensors[2]
        return batch

    def _post_process_action(self, action):
        return (
            ((action + 1) / 2)
            * (self.stats["action_max"] - self.stats["action_min"])
            + self.stats["action_min"]
        )
