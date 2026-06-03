from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi_value import transforms as _transforms
import openpi_value.models.tokenizer as _tokenizer
from openpi_value.models import model as _model
from openpi_value.shared import array_typing as at
from openpi_value.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        policy_seed: int | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        

        self.transforms = transforms
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        self.cfg_scale = getattr(model, 'cfg_scale', 1.0)
        self.set_advantage = getattr(model, 'set_advantage', 1.0)
        
        
        if self.cfg_scale > 1.0:
            self.tokenize_prompt =  _transforms.TokenizePrompt(
                                    _tokenizer.PaligemmaTokenizer(200),  # * Pi05
                                    discrete_state_input=True            # * Pi05
                                )


        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            seed = policy_seed if policy_seed is not None else 0
            logging.info(f"Initializing JAX Policy with SEED: {seed}")

            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            # self._rng = rng or jax.random.key(0)   # * TODO: Customize RNG seeding
            self._rng = rng or jax.random.key(seed)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        
        # cfg_scale = self._model.cfg_scale
        cfg_scale = self.cfg_scale
        
        if 'obs' in obs:
            obs = obs['obs']

        obs_copy = obs.copy()
        
        for cam in obs["images"]:
            obs[f"observation.images.{cam}"] = obs["images"][cam]
        obs.pop("images")

        obs['observation.state'] = obs['state']

        if 'action_advantage' in self.transforms[0].structure:
            # obs['action_advantage'] = torch.tensor(1.0)
            obs['action_advantage'] = torch.tensor(self.set_advantage)

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)  # * go to xxx


        if cfg_scale > 1:
            if 'action_advantage' in self.transforms[0].structure:
                obs_copy['action_advantage'] = torch.tensor(-1.0)  # * -1 for uncond.

            inputs_uncond = inputs.copy()
            updated_prompt = self.tokenize_prompt(obs_copy)
            updated_dict = {
                "tokenized_prompt": updated_prompt["tokenized_prompt"],
                "tokenized_prompt_mask": updated_prompt["tokenized_prompt_mask"],
                "action_advantage": updated_prompt["action_advantage"],
            }
            inputs_uncond.update(updated_dict)

            # * merge inputs and inputs_uncond
            inputs = jax.tree.map(lambda x, y: np.stack([np.array(x), np.array(y)]), inputs, inputs_uncond)

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            if cfg_scale == 1.0:
                inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            else:
                inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        
        # * noise is None here
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
