#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.InternVLA_A1_3B.configuration_internvla_a1 import QwenA1Config
from lerobot.policies.InternVLA_A1_2B.configuration_internvla_a1 import InternA1Config
from lerobot.policies.qwenaction.configuration_qwenaction import QwenActionConfig
from lerobot.policies.WSA_Base.configuration_wsa_base import WSABaseConfig
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.WSA_Large.configuration_wsa_large import WSALargeConfig
from lerobot.policies.names import (
    WSA_BASE,
    WSA_LARGE,
    canonical_policy_type,
    is_wsa_base,
    is_wsa_large,
)
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    Retrieves a policy class by its registered name.

    This function uses dynamic imports to avoid loading all policy classes into memory
    at once, improving startup time and reducing dependencies.

    Args:
        name: The name of the policy. Supported names are "tdmpc", "diffusion", "act",
              "vqbet", "pi0", "pi05", "sac", "reward_classifier", "smolvla".

    Returns:
        The policy class corresponding to the given name.

    Raises:
        NotImplementedError: If the policy name is not recognized.
    """
    name = canonical_policy_type(name)
    if name == "qwena1" or name == "internvla_a1_3b":
        from lerobot.policies.InternVLA_A1_3B.modeling_internvla_a1 import QwenA1Policy

        return QwenA1Policy
    elif name == "qwenaction":
        from lerobot.policies.qwenaction.modeling_qwenaction import QwenActionPolicy

        return QwenActionPolicy
    elif is_wsa_base(name):
        from lerobot.policies.WSA_Base.modeling_wsa_base import WSABasePolicy

        return WSABasePolicy
    elif name == "fastwam":
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

        return FastWAMPolicy
    elif is_wsa_large(name):
        from lerobot.policies.WSA_Large.modeling_wsa_large import WSALargePolicy

        return WSALargePolicy
    
    elif name == "interna1" or name == "internvla_a1_2b":
        from lerobot.policies.InternVLA_A1_2B.modeling_internvla_a1 import InternA1Policy

        return InternA1Policy

    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy

    elif name == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy

    else:
        try:
            return _get_policy_cls_from_policy_name(name=name)
        except Exception as e:
            raise ValueError(f"Policy type '{name}' is not available.") from e


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    Instantiates a policy configuration object based on the policy type.

    This factory function simplifies the creation of policy configuration objects by
    mapping a string identifier to the corresponding config class.

    Args:
        policy_type: The type of the policy. Supported types include "tdmpc",
                     "diffusion", "act", "vqbet", "pi0", "pi05", "sac", "smolvla",
                     "reward_classifier".
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `PreTrainedConfig` subclass.

    Raises:
        ValueError: If the `policy_type` is not recognized.
    """
    policy_type = canonical_policy_type(policy_type)
    if policy_type == "qwena1":
        return QwenA1Config(**kwargs)
    elif policy_type == "qwenaction":
        return QwenActionConfig(**kwargs)
    elif policy_type == WSA_BASE:
        return WSABaseConfig(**kwargs)
    elif policy_type == "fastwam":
        return FastWAMConfig(**kwargs)
    elif policy_type == WSA_LARGE:
        return WSALargeConfig(**kwargs)
    elif policy_type == "internvla_a1":
        return InternA1Config(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi05":
        return PI05Config(**kwargs)
    else:
        try:
            config_cls = PreTrainedConfig.get_choice_class(policy_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Policy type '{policy_type}' is not available.") from e


def make_policy(
    cfg: PreTrainedConfig,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
    Returns:
        An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps').
    """
    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    return policy


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedConfig]:
    """Get policy class from its registered name using dynamic imports.

    This is used as a helper function to import policies from 3rd party lerobot plugins.

    Args:
        name: The name of the policy.
    Returns:
        The policy class corresponding to the given name.
    """
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")  # e.g., DiffusionConfig -> Diffusion
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            f"Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"  # e.g., DiffusionConfig -> DiffusionPolicy
    module_path = config_cls.__module__.replace(
        "configuration_", "modeling_"
    )  # e.g., configuration_diffusion -> modeling_diffusion

    module = importlib.import_module(module_path)
    policy_cls = getattr(module, cls_name)
    return policy_cls
