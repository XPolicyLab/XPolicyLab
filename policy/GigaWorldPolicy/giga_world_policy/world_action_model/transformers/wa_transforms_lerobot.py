import copy
import json
import os
import random
 
import numpy as np
import torch
import torch.nn.functional as torch_F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
 
from giga_train import TRANSFORMS
 
from .wa_transforms import WATransforms
 
 
@TRANSFORMS.register
class WATransformsLerobot(WATransforms):
    def __init__(
        self,
        is_train=False,
        dst_size=None,
        num_frames=1,
        fps=16,
        norm_path=None,
        robotype_to_embed_id=None,
        robotype_default_embed_id=0,
        model_action_dim=None,
        image_cfg=None,
        num_views=1,
        view_keys=None,
        state_key="observation.state",
        action_key="action",
        task_key="task",
        t5_len=64,
    ):
        if norm_path is None:
            raise ValueError("norm_path is None")
        if isinstance(norm_path, (str, os.PathLike)):
            norm_paths = [str(norm_path)]
        else:
            norm_paths = [str(p) for p in norm_path]
            if len(norm_paths) == 0:
                raise ValueError("norm_path list is empty")
 
        super().__init__(
            is_train=is_train,
            dst_size=dst_size,
            num_frames=num_frames,
            fps=fps,
            norm_path=norm_paths[0],
            image_cfg=image_cfg,
            num_views=num_views,
        )
 
        self.robotype_default_embed_id = int(robotype_default_embed_id)
        self.robotype_to_embed_id = dict(robotype_to_embed_id)
        self.model_action_dim = None if model_action_dim is None else int(model_action_dim)
 
        self.norm_paths = norm_paths
        self.stats_dicts = []
        for json_path in self.norm_paths:
            with open(json_path, "r", encoding="utf-8") as f:
                self.stats_dicts.append(json.load(f))
            print("Loading stats dict from:", json_path)
 
        if view_keys is None:
            view_keys = [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ]
        self.view_keys = list(view_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.task_key = task_key
        self.t5_len = int(t5_len)
        self._warned_unknown_robotype = False
 
    def _parse_robotype(self, robotype):
        if robotype is None:
            return None
        if isinstance(robotype, bytes):
            robotype = robotype.decode("utf-8", errors="ignore")
        if hasattr(robotype, "item"):
            try:
                robotype = robotype.item()
            except Exception as e:
                print(f"Error parsing robotype={robotype!r}: {e}")
        if isinstance(robotype, str):
            robotype = robotype.strip()
        return robotype
 
    def _get_robotype_embed_id(self, data_dict) -> int:
        robotype = self._parse_robotype(data_dict.get("robotype", None))
        if robotype in self.robotype_to_embed_id:
            return int(self.robotype_to_embed_id[robotype])
        if isinstance(robotype, str):
            robotype_l = robotype.lower()
            if "agibot" in robotype_l and "agibot" in self.robotype_to_embed_id:
                return int(self.robotype_to_embed_id["agibot"])
            if "aloha" in robotype_l and "aloha" in self.robotype_to_embed_id:
                return int(self.robotype_to_embed_id["aloha"])
            if "agilex" in robotype_l and "agilex" in self.robotype_to_embed_id:
                return int(self.robotype_to_embed_id["agilex"])
        if not self._warned_unknown_robotype:
            print(f"Unknown robotype={robotype!r}, fallback to {self.robotype_default_embed_id}")
            self._warned_unknown_robotype = True
        return self.robotype_default_embed_id
 
    def _get_stats_dict(self, embed_id: int):
        if not self.stats_dicts:
            return self.stats_dict
        if 0 <= embed_id < len(self.stats_dicts):
            return self.stats_dicts[embed_id]
        if not self._warned_unknown_robotype:
            print(f"robotype_embed_id={embed_id} out of range for norm_paths (len={len(self.stats_dicts)}), fallback to 0")
            self._warned_unknown_robotype = True
        return self.stats_dicts[0]
 
    def _to_nchw_uint8(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[0] in (1, 3):
                x = x[None, ...]
            elif x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)[None, ...]
            else:
                x = x[None, ...]
        if x.dim() != 4:
            raise ValueError(f"Unexpected image tensor shape: {tuple(x.shape)}")
        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dtype != torch.uint8:
            x_f = x.to(dtype=torch.float32)
            x_max = float(x_f.max().item()) if x_f.numel() > 0 else 0.0
            if x_max <= 1.0:
                x_f = x_f * 255.0
            x = x_f.clamp(0.0, 255.0).to(dtype=torch.uint8)
        return x
 
    def _sample_frames(self, frames: torch.Tensor, num: int) -> torch.Tensor:
        t = int(frames.shape[0])
        if t <= 0:
            raise ValueError("Empty video frames")
        if t == 1:
            return frames.repeat(num, 1, 1, 1)
        idx = np.linspace(0, t - 1, num=num, dtype=int)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=frames.device)
        return torch.index_select(frames, 0, idx_t)
 
    def _process_images(self, input_images: torch.Tensor, dst_width: int, dst_height: int) -> torch.Tensor:
        input_images = input_images.to(dtype=torch.float32) / 255.0
        height = int(input_images.shape[2])
        width = int(input_images.shape[3])
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        x1 = random.randint(0, new_width - dst_width)
        y1 = random.randint(0, new_height - dst_height)
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
        input_images = self.normalize(input_images)
        return input_images
 
    def __call__(self, data_dict):
        if self.dst_size is None:
            raise ValueError("dst_size is required")
        dst_width, dst_height = self.dst_size
 
        if "robotype" not in data_dict:
            raise KeyError("Missing robotype key")
        robotype_embed_id = self._get_robotype_embed_id(data_dict)
        stats_dict = self._get_stats_dict(robotype_embed_id)
 
        views = []
        for k in self.view_keys[: self.num_views]:
            if k not in data_dict:
                raise KeyError(f"Missing view key: {k}")
            v = data_dict[k]
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Unsupported image type for {k}: {type(v)}")
            v = self._to_nchw_uint8(v)
            v = self._process_images(v, dst_width=dst_width, dst_height=dst_height)
            views.append(v)
 
        if len(views) == 1:
            input_images = views[0]
        else:
            input_images = torch.cat(views, dim=-1)
 
        data_dict["input_images"] = input_images
 
        if self.image_cfg is not None:
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_images"].shape[0])
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            ref_images = data_dict["input_images"].clone() * ref_masks
            data_dict["input_ref_images"] = ref_images
            data_dict["input_ref_masks"] = ref_latent_masks
 
        action = data_dict[self.action_key]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        state = data_dict[self.state_key]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
 
        action = action.to(dtype=torch.float32)
        state = state.to(dtype=torch.float32)

        if action.dim() == 1:
            action = action[None, :]
        if state.dim() == 1:
            state = state[None, :]
        if state.dim() == 2 and state.shape[0] > 1:
            state = state[:1]
 
        if action.shape[0] != self.num_frames:
            t = int(self.num_frames)
            cur_t = int(action.shape[0])
            if cur_t >= t:
                action = action[:t]
            else:
                pad = torch.zeros((t - cur_t, action.shape[1]), dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=0)

        assert self.model_action_dim is not None, "model_action_dim must be provided"
        d = int(self.model_action_dim)
        if state.shape[-1] > d:
            state = state[..., :d]
        if state.shape[-1] < d:
            state = torch_F.pad(state, (0, d - int(state.shape[-1])), value=0.0)
        if action.shape[-1] > d:
            action = action[..., :d]
        if action.shape[-1] < d:
            action = torch_F.pad(action, (0, d - int(action.shape[-1])), value=0.0)
 
        delta_mask_templates = {
            0: np.array([True, True, True, True, True, True, False, True, True, True, True, True, True, False], dtype=bool),
            1: np.array([True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False], dtype=bool),
        }
        base = delta_mask_templates.get(robotype_embed_id, None)
        assert base is not None, f"robotype_embed_id {robotype_embed_id} not found in delta_mask_templates"
        if d > len(base):
            base = np.pad(base, (0, d - len(base)), constant_values=False)
        else:
            base = base[:d]
 
        mask_t = torch.as_tensor(base, dtype=torch.bool, device=action.device)
        idx = torch.nonzero(mask_t, as_tuple=False).flatten()
        delta = action.clone()
        if idx.numel() > 0:
            delta[:, idx] = action[:, idx] - state[:, idx]

        def _to_padded_1d(x, pad_value: float, device):
            t = torch.as_tensor(x, dtype=torch.float32, device=device).flatten()
            if int(t.numel()) >= d:
                return t[:d]
            out = torch.full((d,), float(pad_value), dtype=torch.float32, device=device)
            if int(t.numel()) > 0:
                out[: t.numel()] = t
            return out

        state_mean = _to_padded_1d(stats_dict["norm_stats"]["observation.state"]["mean"], 0.0, state.device)
        state_std = _to_padded_1d(stats_dict["norm_stats"]["observation.state"]["std"], 1.0, state.device)
        delta_mean = _to_padded_1d(stats_dict["norm_stats"]["action"]["mean"], 0.0, action.device)
        delta_std = _to_padded_1d(stats_dict["norm_stats"]["action"]["std"], 1.0, action.device)

        norm_state = state.clone()
        eps = 1e-8
        norm_state = (state - state_mean) / state_std.clamp_min(eps)

        norm_delta = delta.clone()
        norm_delta = (delta - delta_mean) / delta_std.clamp_min(eps)
 
        prompt = data_dict.get("t5_embedding", None)
        assert prompt is not None, "t5_embedding must be provided"
        if isinstance(prompt, np.ndarray):
            prompt = torch.from_numpy(prompt)
        prompt = prompt.to(dtype=torch.float32)
        t5_len = int(self.t5_len)
        prompt = prompt[:t5_len]
        prompt_embeds = torch_F.pad(prompt, (0, 0, 0, t5_len - prompt.shape[0]), value=0)
 
        out = {}
        out["fps"] = torch.tensor(self.fps, dtype=torch.float32)
        out["images"] = data_dict["input_images"]
        out["ref_images"] = data_dict.get("input_ref_images", None)
        out["ref_masks"] = data_dict.get("input_ref_masks", None)
        out["prompt_embeds"] = prompt_embeds
        out["action"] = norm_delta
        out["state"] = norm_state
        out["robotype_embed_id"] = torch.tensor(int(robotype_embed_id), dtype=torch.long)
 
        keys = list(out.keys())
        for k in keys:
            if out[k] is None:
                out.pop(k)
 
        dim = out["action"].shape[-1]
        base = delta_mask_templates.get(robotype_embed_id, None)
        effective_dim = int(np.asarray(base).shape[0]) if base is not None else dim
        effective_dim = min(effective_dim, dim)
        out["action_dim_mask"] = (torch.arange(dim, device=out["action"].device) < effective_dim).to(dtype=torch.bool)
 
        return out
