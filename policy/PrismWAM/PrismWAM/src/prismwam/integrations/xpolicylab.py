"""XPolicyLab lifecycle backed by the public PrismWAM deployment API.

The server owns image decoding. This adapter owns one native-action history per
environment and returns physical joint commands through XPolicyLab's shared
state packing helpers. No simulator or robot execution occurs here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from prismwam.deployment.policy import ActionStatistics, load_policy, read_config
from prismwam.deployment.history import RGBHistoryBuffer
from prismwam.hub import bundle_files
from .observations import Preprocessor


SUPPORTED_RECIPES = {
    "robodojo",
    "robotwin",
    "realworld",
}


class PrismWAMAdapter:
    """Implement XPolicyLab's single and batched observation/action contract.

    Batched environments share model weights but run inference sequentially.
    Their histories and instructions are independent. ``policy`` and
    ``robot_info`` allow callers to supply an already loaded policy and robot
    metadata; production wrappers obtain metadata from XPolicyLab itself.
    """

    def __init__(self, usr_args, *, policy=None, robot_info=None):
        self.args = copy.deepcopy(dict(usr_args))
        self.recipe = self.args.get("recipe")
        if self.recipe not in SUPPORTED_RECIPES:
            raise ValueError(
                "Set recipe to a supported RoboDojo, RoboTwin or real-world recipe. "
                "LIBERO's 7D delta end-effector actions are not XPolicyLab's "
                "absolute 7D pose contract and require its native adapter."
            )
        if self.args.get("action_type") != "joint":
            raise ValueError("These checkpoints require action_type=joint.")
        from XPolicyLab.utils.process_data import (
            get_robot_action_dim_info,
            pack_robot_state,
            unpack_robot_state,
        )

        self._pack_state = pack_robot_state
        self._unpack_action = unpack_robot_state
        self.robot_info = copy.deepcopy(
            robot_info
            if robot_info is not None
            else get_robot_action_dim_info(self.args["env_cfg_type"])
        )
        arms, grippers = self.robot_info["arm_dim"], self.robot_info["ee_dim"]
        if (
            len(arms) not in (1, 2)
            or len(arms) != len(grippers)
            or any(
                not isinstance(d, int) or isinstance(d, bool) or d < 1
                for d in arms + grippers
            )
        ):
            raise ValueError("Invalid robot action dimensions from XPolicyLab.")
        self.native_dimension = sum(arms) + sum(grippers)
        realworld = self.recipe == "realworld"
        if not realworld and (list(arms) != [6, 6] or list(grippers) != [1, 1]):
            raise ValueError(
                "This recipe requires two six-joint arms with one gripper each."
            )
        if realworld and self.native_dimension not in (7, 14):
            raise ValueError(
                "The real-world recipe supports native 7D or 14D joint commands."
            )
        seed = self.args.get("inference_seed")
        if seed is None:
            seed = self.args.get("seed")
        self.inference_seed = int(0 if seed is None else seed)
        self.raw_statistics = None
        if policy is None:
            policy = self._load_bundle(realworld)
        self.policy = policy
        self.config = copy.deepcopy(dict(policy.config))
        if self.policy.action_dim != 14 or self.policy.proprio_dim != 14:
            raise ValueError("XPolicyLab joint recipes require a 14D model interface.")
        if self.raw_statistics is None and self.policy.statistics is None:
            raise ValueError(
                "Deployment requires physical-unit State/Action statistics."
            )
        if realworld and self.native_dimension == 7 and self.raw_statistics is None:
            raise ValueError(
                "Single-arm deployment requires task-specific native 7D statistics."
            )
        self.preprocessor = Preprocessor(
            self.config,
            self.recipe,
            camera_map=self.args.get("camera_map"),
            depth_scale=self.args.get("depth_scale", 1.0),
            depth_key=self.args.get("depth_key", "depth"),
            task_name=self.args.get("task_name"),
        )
        self.train_config = self.preprocessor.train_config
        self.histories: dict[int, RGBHistoryBuffer] = {}
        self.observations: dict[int, dict[str, Any]] = {}
        self.active_env_ids: list[int] = []
        self._text_contexts: dict[str, dict[str, Any]] = {}
        self._single_env_id = 0
        self.reset()

    def _load_bundle(self, realworld):
        from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

        policy_dir = Path(self.args.get("policy_dir", ".")).expanduser().resolve()
        root = resolve_checkpoint_root(
            self.args,
            policy_dir / "checkpoints",
            policy_dir=policy_dir,
            explicit_keys=("model_path", "checkpoint_path"),
        ).resolve()
        config_path, checkpoint = bundle_files(root)
        if not config_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(
                f"Not a PrismWAM model folder (config.yaml + policy.pt): {root}"
            )
        config = read_config(config_path)
        evaluation = {
            key: value
            for key, value in config.get("deployment", {}).items()
            if key
            in (
                "benchmark",
                "action_horizon",
                "replan_steps",
                "num_inference_steps",
                "inference_shifts",
            )
        }
        legacy_evaluation = (
            root / "evaluation.json"
        )  # folders exported with manifest.json
        if legacy_evaluation.is_file():
            evaluation.update(json.loads(legacy_evaluation.read_text()))
        benchmark = self.recipe
        if str(evaluation.get("benchmark", "")).lower() != benchmark:
            raise ValueError(
                "The model's benchmark does not match the selected recipe."
            )
        expected_replan = 16 if self.recipe == "robodojo" else 24
        if (
            evaluation.get("action_horizon") != 32
            or evaluation.get("replan_steps") != expected_replan
            or evaluation.get("num_inference_steps") != 10
        ):
            raise ValueError(
                "The model's inference settings differ from the selected recipe."
            )
        if realworld:
            selected_task = config.get("deployment", {}).get("task_name")
            if selected_task != self.args.get("task_name") or not selected_task:
                raise ValueError(
                    "Export this real-world model with its explicit task_name."
                )
            if (
                config.get("data", {}).get("train", {}).get("action_dim")
                != self.native_dimension
            ):
                raise ValueError(
                    "The model's task action_dim does not match the robot dimensions."
                )
            stats_path = config["deployment"].pop("statistics_path", None)
            if stats_path is None:
                raise ValueError("Real-world models require task-specific statistics.")
            self.raw_statistics = ActionStatistics.load(
                stats_path, self.native_dimension, expected_task=selected_task
            )
        dtype_name = str(self.args.get("dtype", "bfloat16"))
        if dtype_name not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32.")
        return load_policy(
            config,
            checkpoint,
            device=self.args.get("device", "cuda:0"),
            dtype=getattr(torch, dtype_name),
            evaluation=evaluation,
        )

    def _language(self, instruction: str) -> dict[str, Any]:
        template = self.train_config.get("prompt_template", "{task}")
        prompt = template.format(task=instruction)
        if prompt in self._text_contexts:
            return self._text_contexts[prompt]
        cache_path = self.config.get("deployment", {}).get("text_cache_path")
        if not cache_path:
            if self.recipe == "realworld":
                raise ValueError(
                    "Real-world deployment requires its training text cache to "
                    "preserve the original language attention mask."
                )
            return {"prompt": prompt}
        cache_path = Path(cache_path)
        length = int(self.train_config.get("context_len", 128))
        encoder = self.train_config.get("text_encoder_id", "wan22ti2v5b")
        if cache_path.is_dir():
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            cache_path = cache_path / f"{digest}.t5_len{length}.{encoder}.pt"
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not Path(self.config["deployment"]["text_cache_path"]).is_dir():
            if payload.get("prompt") != prompt:
                raise ValueError(
                    "A single-file text cache must contain its matching prompt."
                )
        context = torch.as_tensor(payload["context"]).float()
        mask = torch.as_tensor(payload.get("mask", payload.get("context_mask"))).bool()
        if (
            context.ndim != 2
            or context.shape[0] != length
            or mask.shape != (length,)
            or not torch.isfinite(context).all()
        ):
            raise ValueError("Invalid cached language context shape or values.")
        # Simulation readers zero padded features and expose all positions;
        # the real-world reader retains the original text attention mask.
        context = context.clone()
        if self.recipe != "realworld":
            context[~mask] = 0
            mask = torch.ones_like(mask)
        result = {"context": context[None], "context_mask": mask[None]}
        self._text_contexts[prompt] = result
        return result

    def _prepare_observation(self, obs: Mapping[str, Any]):
        rgb, depth = self.preprocessor.prepare(obs)
        instruction = obs.get("instruction", obs.get("instructions"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Observation must provide a nonempty instruction string.")
        state = np.asarray(
            self._pack_state(obs, "joint", self.robot_info, source_type="obs"),
            dtype=np.float32,
        )
        if state.shape != (self.native_dimension,) or not np.isfinite(state).all():
            raise ValueError("Expected one finite native robot-state vector.")
        proprio = torch.from_numpy(state.copy())
        if self.raw_statistics is not None:
            proprio = self.raw_statistics.normalize_state(proprio)
            if self.native_dimension < self.policy.proprio_dim:
                proprio = torch.nn.functional.pad(
                    proprio, (0, self.policy.proprio_dim - self.native_dimension)
                )
        prepared = {
            "input_image": depth[None],
            "rgb_input_image": rgb[None],
            "proprio": proprio[None],
            **self._language(instruction),
        }
        return rgb, prepared

    def _store_observation(self, rgb, prepared, env_id):
        history = self.histories.setdefault(
            env_id, RGBHistoryBuffer(self.preprocessor.rgb_history)
        )
        history.append(rgb, step=history.step + 1)
        self.observations[env_id] = prepared

    def update_obs(self, obs):
        """Consume one already decoded observation at an executed-action boundary."""
        env_id = int(obs.get("env_idx", 0))
        self._store_observation(*self._prepare_observation(obs), env_id)
        self._single_env_id = env_id

    def update_obs_batch(self, obs_list):
        """Update only the supplied active environments without discarding others."""
        if not obs_list:
            raise ValueError("update_obs_batch requires a nonempty observation list.")
        ids = [int(obs["env_idx"]) for obs in obs_list]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate env_idx values in an observation batch.")
        # Validate the whole batch before advancing any history. A malformed
        # later observation must not make a retry duplicate earlier frames.
        prepared = [self._prepare_observation(obs) for obs in obs_list]
        for (rgb, observation), env_id in zip(prepared, ids):
            self._store_observation(rgb, observation, env_id)
        self.active_env_ids = ids

    def _get_action(self, env_id):
        if env_id not in self.observations:
            raise RuntimeError("Call update_obs before requesting actions.")
        inputs = dict(self.observations[env_id])
        inputs["rgb_history_images"] = self.histories[env_id].tensor()
        # All retained memory modes are reconstructed from explicit observed
        # frames. Clear per-forward caches before switching environments.
        self.policy.reset()
        prediction = self.policy.predict(inputs, seed=self.inference_seed)
        actions = torch.as_tensor(prediction["action"]).detach().float().cpu()
        if actions.shape != (self.policy.action_horizon, self.policy.action_dim):
            raise ValueError("Policy returned an invalid action chunk shape.")
        if self.raw_statistics is not None:
            if not prediction["normalized"]:
                raise ValueError("Task-specific statistics would be applied twice.")
            actions = self.raw_statistics.denormalize_action(
                actions[:, : self.native_dimension]
            )
        elif prediction["normalized"]:
            raise ValueError("Normalized actions cannot be sent to a robot.")
        if not torch.isfinite(actions).all():
            raise ValueError("Policy returned nonfinite physical actions.")
        if prediction["replan_steps"] != self.policy.replan_steps:
            raise ValueError("Policy changed the recorded replanning interval.")
        return self._unpack_action(
            actions[: self.policy.replan_steps].numpy(),
            "joint",
            self.robot_info,
            source_type="obs",
        )

    def get_action(self):
        return self._get_action(self._single_env_id)

    def get_action_batch(self, env_idx_list=None):
        ids = self.active_env_ids if env_idx_list is None else list(env_idx_list)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Expected nonempty unique active environment indices.")
        return [self._get_action(int(env_id)) for env_id in ids]

    def reset(self, env_id=None):
        """Clear histories at episode boundaries; optionally clear one environment."""
        if env_id is None:
            self.histories.clear()
            self.observations.clear()
            self.active_env_ids.clear()
            self._single_env_id = 0
        else:
            env_id = int(env_id)
            self.histories.pop(env_id, None)
            self.observations.pop(env_id, None)
            self.active_env_ids = [i for i in self.active_env_ids if i != env_id]
        self.policy.reset()

    def prepare_case(self, case_meta=None):
        """Start a new trial; scene/layout seeds never replace the diffusion seed."""
        self.reset()
