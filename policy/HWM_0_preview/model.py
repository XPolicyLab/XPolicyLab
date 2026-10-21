"""XPolicyLab boundary: observations in, absolute joint-action chunks out."""

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_batch_size, get_robot_action_dim_info

from .hwm.config import ModelConfig, bundle_path
from .hwm.model.denoising import draw_noise, sample_actions
from .hwm.model.memory import EpisodeMemory, composite_image, encode_histories
from .hwm.model.network import HWM0Preview
from .hwm.model.planner import Planner

logger = logging.getLogger(__name__)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        if model_cfg["action_type"] != "joint":
            raise ValueError(
                "HWM_0_preview supports absolute joint actions, not end-effector poses"
            )
        self.inference_batch_size = model_cfg.get("inference_batch_size", 4)
        if type(self.inference_batch_size) is not int or self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be a positive integer")
        allowed = {
            "policy_name",
            "protocol",
            "host",
            "port",
            "bench_name",
            "task_name",
            "ckpt_name",
            "env_cfg_type",
            "seed",
            "action_type",
            "gpu_id",
            "eval_batch",
            "num_inference_steps",
            "flow_shift",
            "replan_steps",
            "inference_batch_size",
            "attention_backend",
            "vision_encode_batch_size",
            "use_kv_cache",
            "camera_mapping",
            "model_path",
            "checkpoint_path",
            "ckpt_path",
            "model_dir",
            "pretrained_path",
            "ws_ping_interval_s",
            "ws_ping_timeout_s",
            "request_timeout_s",
            "relay_url",
        }
        unknown = set(model_cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported deployment overrides: {sorted(unknown)}; model semantics belong to the bundle config"
            )
        policy_dir = Path(__file__).resolve().parent
        root = resolve_checkpoint_root(model_cfg, policy_dir / "checkpoints")
        assets = root.parent if root.is_file() else root
        self.config = ModelConfig.load(assets / "config.yaml")
        cfg = self.config
        resolved = cfg.resolve_inference(model_cfg)
        self.num_inference_steps = resolved["num_inference_steps"]
        self.flow_shift = resolved["flow_shift"]
        self.replan_steps = resolved["replan_steps"]
        self.use_kv_cache = model_cfg.get("use_kv_cache", True)
        if type(self.use_kv_cache) is not bool:
            raise ValueError("use_kv_cache must be a boolean")
        mapping = model_cfg.get(
            "camera_mapping",
            {
                "head": "cam_head",
                "left_wrist": "cam_left_wrist",
                "right_wrist": "cam_right_wrist",
            },
        )
        if (
            not isinstance(mapping, dict)
            or set(mapping) != {"head", "left_wrist", "right_wrist"}
            or any(not isinstance(v, str) or not v for v in mapping.values())
            or len(set(mapping.values())) != 3
        ):
            raise ValueError(
                "camera_mapping must map three distinct camera fields to head/left_wrist/right_wrist"
            )
        self.cameras = [mapping[key] for key in ("head", "left_wrist", "right_wrist")]
        info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if len(info["arm_dim"]) != 2 or len(info["ee_dim"]) != 2:
            raise ValueError("HWM_0_preview requires a dual-arm robot")
        self.layout = [
            (f"{side}_{part}_joint_state", int(dims[i]))
            for i, side in enumerate(("left", "right"))
            for part, dims in (("arm", info["arm_dim"]), ("ee", info["ee_dim"]))
        ]
        self.action_dim = sum(size for _, size in self.layout)
        if self.action_dim != cfg.m["action_dim"]:
            raise ValueError(
                "Robot action layout disagrees with bundle action dimension"
            )
        self.batch_size = get_batch_size(model_cfg["env_cfg_type"])
        checkpoint = (
            root if root.is_file() else bundle_path(assets, cfg.assets["weights"])
        )
        paths = {
            name: bundle_path(assets, cfg.assets[name])
            for name in ("statistics", "processor", "tokenizer")
        }
        for name, path in {"checkpoint": checkpoint, **paths}.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing HWM {name}: {path}")
        with paths["statistics"].open() as handle:
            statistics = json.load(handle)[cfg.normalization["key"]]
        self.statistics = {}
        for kind in ("state", "action"):
            mean, std = (
                np.asarray(statistics[kind][key], dtype=np.float32)
                for key in ("mean", "std")
            )
            if mean.shape != (self.action_dim,) or std.shape != mean.shape:
                raise ValueError(f"{kind} statistics do not match robot dimensions")
            if (
                not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or (std < 0).any()
            ):
                raise ValueError(f"Invalid {kind} statistics")
            self.statistics[kind] = mean, std
        processor = AutoProcessor.from_pretrained(
            paths["processor"], local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            paths["tokenizer"], local_files_only=True
        )
        self.model = HWM0Preview.load(
            checkpoint,
            config=cfg,
            attention_backend=model_cfg.get("attention_backend", "flash_attention_2"),
        )
        if self.action_dim != self.model.action.head.out_features:
            raise ValueError("Robot action layout does not match the model")
        self.planner = Planner(
            self.model,
            processor,
            vision_encode_batch_size=model_cfg.get("vision_encode_batch_size", 128),
        )
        self.reset()
        seed = int(model_cfg["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(
            "HWM_0_preview loaded; denoising=%d, flow_shift=%g, inference_batch_size=%d, horizon=%d, execute=%d",
            self.num_inference_steps,
            self.flow_shift,
            self.inference_batch_size,
            cfg.m["action_horizon"],
            self.replan_steps,
        )

    def get_policy_metadata(self):
        """Public preprocessing contract; contains no filesystem paths or secrets."""
        return {
            "protocol": 2,
            "config_hash": self.config.fingerprint,
            "observation": dict(self.config.observation.data),
            "cameras": list(self.cameras),
            "replan_steps": self.replan_steps,
        }

    def reset(self):
        self.episodes = {}
        self.single_env_idx = None

    def update_obs(self, obs):
        env_idx = int(obs.get("env_idx", 0))
        self.single_env_idx = env_idx
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        seen = set()
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            if env_idx in seen:
                raise ValueError(f"Duplicate environment observation: {env_idx}")
            seen.add(env_idx)
            state = self._state(obs)
            image, head = composite_image(
                obs["vision"], self.config.observation, self.cameras
            )
            episode = self.episodes.setdefault(
                env_idx, EpisodeMemory(self.config.observation)
            )
            episode.observe(
                obs.get("instruction", obs.get("instructions")),
                image,
                head,
                state,
            )

    def _state(self, obs):
        values = []
        for key, size in self.layout:
            value = np.asarray(obs["state"][key], dtype=np.float32)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid state {key}: expected finite [{size}]")
            values.append(value)
        return np.concatenate(values)

    @torch.no_grad()
    def infer_context(self, observations):
        """Atomic inference using only request-owned episode/semantic context.

        Official WebSocket dispatch serializes this complete call. No legacy
        `self.episodes` state is consulted or modified. RNG remains process-global.
        """
        prepared = self._prepare_context(observations)
        return {
            "request_id": prepared["request_id"],
            "items": self._predict_context_rows(prepared["rows"]),
        }

    def _prepare_context(self, observations):
        """Validate one request and prepare CPU inputs, without consuming RNG."""
        if not isinstance(observations, list) or not observations:
            raise ValueError("Expected a nonempty list of context observations")
        indices, episodes, histories, metadata = [], [], [], []
        request_id = None
        episode_ids = set()
        for obs in observations:
            index = obs["env_idx"]
            if type(index) is not int or index < 0 or index in indices:
                raise ValueError("Expected unique nonnegative environment indices")
            context = obs["hwm_context"]
            if (
                context.get("protocol") != 2
                or context.get("config_hash") != self.config.fingerprint
            ):
                raise ValueError(
                    "HWM context protocol/config does not match this server"
                )
            for key in ("request_id", "episode_id", "semantic", "subtask"):
                if not isinstance(context.get(key), str) or not context[key].strip():
                    raise ValueError(f"Invalid context {key}")
            if request_id is None:
                request_id = context["request_id"]
            if (
                context["request_id"] != request_id
                or context["episode_id"] in episode_ids
            ):
                raise ValueError("Mixed request IDs or duplicate episode IDs")
            episode_ids.add(context["episode_id"])
            for key in ("step", "state_version"):
                if type(context.get(key)) is not int or context[key] < 0:
                    raise ValueError(f"Invalid context {key}")
            if type(context.get("cache_valid")) is not bool:
                raise ValueError("Invalid context cache_valid")
            step = context["step"]
            offsets = self.config.observation.offsets(step)
            if context.get("history_offsets") != offsets:
                raise ValueError(
                    "History offsets do not match the control-step contract"
                )
            frames = np.asarray(context["history_images"])
            if frames.dtype != np.uint8 or frames.shape != (
                len(offsets),
                self.config.observation.height,
                self.config.observation.width,
                3,
            ):
                raise ValueError(
                    "History image shape/dtype disagrees with bundle config"
                )
            episode = EpisodeMemory(self.config.observation)
            episode.instruction = obs.get("instruction", obs.get("instructions"))
            if (
                not isinstance(episode.instruction, str)
                or not episode.instruction.strip()
            ):
                raise ValueError("Missing task instruction")
            episode.current, _ = composite_image(
                obs["vision"], self.config.observation, self.cameras
            )
            episode.state = self._state(obs)
            episode.step = step
            episode.semantic, episode.subtask = context["semantic"], context["subtask"]
            episode.cache_valid = context["cache_valid"]
            indices.append(index)
            episodes.append(episode)
            histories.append((list(frames), offsets))
            metadata.append(
                {
                    "env_idx": index,
                    "episode_id": context["episode_id"],
                    "step": step,
                    "state_version": context["state_version"] + 1,
                }
            )
        return {
            "request_id": request_id,
            "rows": list(zip(episodes, histories, metadata, strict=True)),
        }

    @torch.no_grad()
    def _predict_context_rows(self, rows):
        """Predict rows from any clients; metadata remains attached to each row."""
        episodes, histories, metadata = zip(*rows, strict=True)
        actions = self._predict(
            episodes, histories, [meta["env_idx"] for meta in metadata]
        )
        return [
            {
                **meta,
                "actions": chunk,
                "semantic": episode.semantic,
                "subtask": episode.subtask,
                "cache_valid": episode.cache_valid,
            }
            for meta, chunk, episode in zip(metadata, actions, episodes, strict=True)
        ]

    def get_action(self):
        if self.single_env_idx is None:
            raise RuntimeError("Call update_obs before get_action")
        return self.get_action_batch([self.single_env_idx])[0]

    @torch.no_grad()
    def get_action_batch(self, env_idx_list=None):
        indices = (
            list(range(self.batch_size)) if env_idx_list is None else list(env_idx_list)
        )
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate active environment indices")
        if any(i not in self.episodes for i in indices):
            raise ValueError("Missing environment observation")
        episodes = [self.episodes[i] for i in indices]
        return self._predict(
            episodes, [episode.history() for episode in episodes], indices
        )

    def _predict(self, all_episodes, all_histories, indices):
        result = []
        for start in range(0, len(indices), self.inference_batch_size):
            active = indices[start : start + self.inference_batch_size]
            episodes = all_episodes[start : start + self.inference_batch_size]
            histories = all_histories[start : start + self.inference_batch_size]
            requests = [
                {
                    "instruction": e.instruction,
                    "memory": e.semantic,
                    "subtask": e.subtask,
                    "cache_valid": e.cache_valid,
                    "current": e.current,
                    "history": h[0],
                }
                for e, h in zip(episodes, histories, strict=True)
            ]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                plans = self.planner.batch(requests)
                device = self.model.world_queries.device
                text, masks = self.model.encode_video_text(
                    self.tokenizer,
                    [e.instruction for e in episodes],
                    [p[4] for p in plans],
                )
                pixels = (
                    torch.from_numpy(np.stack([e.current for e in episodes]))
                    .float()
                    .div(255)
                    .mul(2)
                    .sub(1)
                )
                pixels = pixels.permute(0, 3, 1, 2)[:, :, None].to(
                    device=device, dtype=torch.bfloat16
                )
                first = self.model.encode_video(pixels)
                history_latents = encode_histories(
                    self.model, [e.current for e in episodes], histories
                )
                mean, std = self.statistics["state"]
                states = torch.from_numpy(
                    np.stack(
                        [
                            np.clip(
                                (e.state - mean)
                                / (std + self.config.normalization["epsilon"]),
                                *self.config.normalization["state_clip"],
                            )
                            for e in episodes
                        ]
                    )
                )
                noises = [
                    draw_noise(first[i : i + 1], self.config)
                    for i in range(len(active))
                ]
                groups = {}
                for i, h in enumerate(history_latents):
                    groups.setdefault(
                        None if h is None else tuple(h[0].shape), []
                    ).append(i)
                normalized = [None] * len(active)
                for members in groups.values():
                    history = (
                        None
                        if history_latents[members[0]] is None
                        else tuple(
                            torch.stack([history_latents[i][j] for i in members])
                            for j in range(2)
                        )
                    )
                    args = (
                        self.model,
                        first[members],
                        text[members],
                        masks[members],
                        torch.cat([plans[i][0] for i in members]),
                        torch.cat([plans[i][1] for i in members]),
                        states[members],
                        history,
                    )
                    noise = tuple(
                        torch.cat([noises[i][j] for i in members]) for j in range(2)
                    )
                    values, _ = sample_actions(
                        *args,
                        num_inference_steps=self.num_inference_steps,
                        flow_shift=self.flow_shift,
                        use_kv_cache=self.use_kv_cache,
                        noise=noise,
                    )
                    for row, i in enumerate(members):
                        normalized[i] = values[row]
            mean, std = self.statistics["action"]
            values = (
                torch.stack(normalized).float().cpu().numpy()
                * (std + self.config.normalization["epsilon"])
                + mean
            )
            if not np.isfinite(values).all():
                raise ValueError("Policy generated nonfinite actions")
            for index, e, p, env_values in zip(
                active, episodes, plans, values, strict=True
            ):
                chunk = []
                for row in env_values[: self.replan_steps]:
                    command = {}
                    cursor = 0
                    for key, size in self.layout:
                        command[key] = row[cursor : cursor + size].copy()
                        cursor += size
                    chunk.append(command)
                e.commit(*p[2:])
                logger.info(
                    "HWM env=%s step=%d decision=%s subtask=%r",
                    index,
                    e.step,
                    p[2],
                    p[4],
                )
                result.append(chunk)
        return result
