"""Fixed-shape, genuinely batched Pi05 inference for EchoPolicy candidates."""
from pathlib import Path
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from XPolicyLab.policy.Pi_05.model import Model as Pi05Model, encode_obs
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_action_dim, unpack_robot_state

logger = logging.getLogger(__name__)


class Model(Pi05Model):
    def __init__(self, model_cfg):
        logging.basicConfig(force=True, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if model_cfg.get("env_cfg_type") != "arx_x5" or model_cfg.get("action_type", "joint") != "joint":
            raise ValueError("EchoPolicy supports env_cfg_type=arx_x5 and action_type=joint")
        self.batch_size = int(model_cfg.get("batch_size", 160))
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        model_cfg = dict(model_cfg)
        policy_dir = Path(__file__).resolve().parent
        model_cfg["checkpoint_path"] = str(resolve_checkpoint_root(
            model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir))
        super().__init__(model_cfg)
        self._observations = []
        if model_cfg.get("warmup", True):
            sample = {
                "state": np.zeros(get_action_dim(model_cfg["env_cfg_type"]), dtype=np.float32),
                "images": {name: np.zeros((3, 224, 224), dtype=np.uint8) for name in
                           ("cam_high", "cam_left_wrist", "cam_right_wrist")},
                "instruction": "Cover the blocks with the matching cups.",
            }
            self.infer_batch([sample])
            logger.info("Pi05 warmup complete: batch_size=%d", self.batch_size)

    def update_obs_batch(self, obs_list):
        self._observations = list(obs_list)
        self._latest_env_idx_list = [obs.get("env_idx", i) for i, obs in enumerate(self._observations)]
        if len(set(self._latest_env_idx_list)) != len(self._latest_env_idx_list):
            raise ValueError("Duplicate environment indices")

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._observations:
            raise RuntimeError("update_obs or update_obs_batch must precede action retrieval")
        indices = self._latest_env_idx_list if env_idx_list is None else env_idx_list
        by_env = dict(zip(self._latest_env_idx_list, self._observations, strict=True))
        return self.infer_batch([by_env[index] for index in indices])["actions"]

    def infer_batch(self, observations):
        """Atomic RPC: preprocess individually, stack and pad, run one JAX batch."""
        if not observations:
            return {"actions": []}
        results = []
        start = time.perf_counter()
        for offset in range(0, len(observations), self.batch_size):
            chunk = observations[offset:offset + self.batch_size]
            transformed = []
            for obs in chunk:
                # EchoPolicy conditions Pi05 on the original instruction. Its
                # default CFG branches are identical, so guidance cancels out.
                uncond = obs.get("cfg_uncond_prompt", obs.get("instruction"))
                if obs.get("cfg_scale") is not None and uncond != obs.get("instruction"):
                    raise ValueError("This Pi05 adapter requires identical CFG prompts")
                transformed.append(self.policy._input_transform(
                    encode_obs(obs, self.action_type, self.robot_action_dim_info)))
            transformed += [transformed[-1]] * (self.batch_size - len(transformed))
            inputs = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *transformed)
            self.policy._rng, rng = jax.random.split(self.policy._rng)
            actions = np.asarray(self.policy._sample_actions(
                rng, model_lib.Observation.from_dict(inputs), **self.policy._sample_kwargs))
            # Output transforms operate on single examples (state-relative
            # actions and robot gripper conversion), as during training.
            for index in range(len(chunk)):
                output = self.policy._output_transform({
                    "state": np.asarray(inputs["state"][index]), "actions": actions[index]})
                action = output["actions"]
                if self.robot_action_dim_info is not None:
                    action = unpack_robot_state(action, self.action_type,
                                               self.robot_action_dim_info, source_type="obs")
                results.append(action)
        logger.info("Pi05 candidates=%d batch_size=%d latency=%.3fs", len(observations),
                    self.batch_size, time.perf_counter() - start)
        return {"actions": results}

    def reset(self):
        self._observations = []
        super().reset()
