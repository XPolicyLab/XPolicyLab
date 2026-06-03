import argparse
import os
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import megfile
from loguru import logger

from dexbotic.data.dataset.transform.action import (ActionNormAnd2String,
                                                    AddAction, DeltaAction, PadAction, PadState)
from dexbotic.data.dataset.transform.common import (Pipeline, ToDict, ToList,
                                                    ToNumpy)
from dexbotic.data.dataset.transform.language import (AddPromptTemplate,
                                                      ReplaceAnswer)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.base_exp import ComputeNormActionConfig as _ComputeNormActionConfig
from dexbotic.exp.cogact_exp import CogACTActionConfig as _CogACTActionConfig
from dexbotic.exp.cogact_exp import CogACTDataConfig as _CogACTDataConfig
from dexbotic.exp.cogact_exp import CogACTExp as _CogACTExp
from dexbotic.exp.cogact_exp import CogACTModelConfig as _CogACTModelConfig
from dexbotic.exp.cogact_exp import CogACTTrainerConfig as _CogACTTrainerConfig
from dexbotic.exp.cogact_exp import InferenceConfig as _CogACTInferenceConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=[
            'train',
            'inference'])
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class RoboTwin2CogActTrainerConfig(_CogACTTrainerConfig):
    """
    Trainer configuration class - controls training process parameters

    Modification instructions:
    - output_dir: Training output directory, recommended to modify to your local path
    - wandb_project: Weights & Biases project name for experiment tracking
    - num_train_epochs: Number of training epochs, adjust based on dataset size
    """
    # NOTE: Training output directory - modify to your local path, format: /path/to/your/output/directory
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/robotwin2_cogact/adjust_bottle-{datetime.now().strftime("%m%d")}')
    # Weights & Biases project name - for experiment tracking and visualization
    wandb_project: str = field(default='dexbotic_robotwin2_cogact')
    # NOTE: Adjust the number of iterations to about 30,000
    num_train_epochs: int = field(default=550)


class AddRelativeTrajectory:
    def __init__(self,
                 trajectory_length: int = 10,
                 padding_mode: str = 'last',
                 padding_action: bool = False,
                 ):
        """Args:
            trajectory_length: int, the length of the trajectory. Default: 10
            padding_mode: str, the padding mode for the trajectory. Default: 'last'
            padding_action: bool, whether to pad the action if the length of the action is less than the trajectory length. Default: False
        """
        self.trajectory_length = trajectory_length
        self.padding_mode = padding_mode
        self.padding_action = padding_action
        assert self.padding_mode in ['last', 'zero'], 'only support `last` and `zero` padding mode in constructing trajectory'
    
    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if 'action' not in episode_data_dict:
            # warnings.warn('action is not in the episode_data_dict, skip the AddTrajectory transform')
            return episode_data_dict

        episode_data_dict['meta_data']['trajectory_length'] = self.trajectory_length
        gripper_dim = episode_data_dict['meta_data'].get('gripper_dim', [-1])
        
        action  = episode_data_dict['action'] # shape: N D
        valid_trajectory_length = len(action)
        
        if self.padding_action:
            action = self.pad(action, self.trajectory_length, gripper_dim)
        else:
            assert len(action) >= self.trajectory_length, f'the length of the action in {episode_data_dict["meta_data"].get("jsonl_file", "Unkown")} should be larger than the trajectory length'
        
        trajectory = [action]
        for i in range(1, self.trajectory_length):
            _next_action = np.copy(action[i:])
            _next_action = self.pad(_next_action, len(action), gripper_dim)
            _next_action = self.action_add(trajectory[-1], _next_action, gripper_dim)
        
            trajectory.append(_next_action)
        trajectory = np.stack(trajectory, axis=-1) # shape: N D T
        # reshape to N T D than N (T * D)
        trajectory = np.transpose(trajectory, (0, 2, 1)).reshape(trajectory.shape[0], -1)
        trajectory = trajectory[:valid_trajectory_length]
        episode_data_dict['trajectory'] = trajectory
        episode_data_dict['action'] = trajectory
        return episode_data_dict
    
    def pad(self, action, trajectory_length, gripper_dim):
        if len(action) >= trajectory_length:
            return action
        else:
            if self.padding_mode == 'zero':
                padding_action = np.zeros_like(action[-1])
                padding_action[gripper_dim] = action[-1][gripper_dim]
                
            else:
                padding_action = action[-1]
        action = np.concatenate([action,
                                 np.array([np.copy(padding_action) for _ in range(trajectory_length - len(action))])], axis=0)
        return action

    def action_add(self, action1, action2, gripper_dim):
        """Add two actions, keeping the gripper dimension of action2."""
        new_action = action1 + action2
        new_action[:, gripper_dim] = action2[:, gripper_dim]
        return new_action


@dataclass
class RoboTwin2CogActActionConfig(_CogACTActionConfig):
    """
    Action configuration class - controls action-related parameters
    """
    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline([
            ToDict(),
            ToNumpy(),
            AddAction(predict_length=1),
            PadState(ndim=16, axis=-1),
            PadAction(ndim=16, axis=-1),
            DeltaAction(enable=self.delta),
            AddRelativeTrajectory(
                trajectory_length=self.trajectory_length,
                padding_mode=self.trajectory_padding_model,
                padding_action=self.padding_action
            ),
            ActionNormAnd2String(
                statistic_mapping=statistic_mapping,
                vocab_size=self.vocab_size,
                string_format=self.string_format
            ),
            LoadMultiModal(),
            AddPromptTemplate(prompt_template=self.prompt_template),
            ReplaceAnswer(default_answer=self.replace_with_default_answer),
            ToList(),
        ])

        return action_config


@dataclass
class RoboTwin2CogActComputeNormActionConfig(_ComputeNormActionConfig):
    
    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline([
            ToDict(),
            ToNumpy(),
            AddAction(predict_length=16),
            PadState(ndim=16, axis=-1),
            PadAction(ndim=16, axis=-1),
            DeltaAction(enable=self.delta),
            ToList(),
        ])

        return action_config


@dataclass
class RoboTwin2CogActDataConfig(_CogACTDataConfig):
    """
    Data configuration class - controls dataset and data processing parameters

    Modification instructions:
    - dataset_name: Dataset name, needs to match your dataset
    - action_config: Action configuration instance
    """
    # NOTE: Dataset name - modify to task name you want to train
    dataset_name: str = field(default='robotwin2_adjust_bottle')
    num_images: int = field(default=3)
    # Action configuration - contains action-related parameter settings
    action_config: RoboTwin2CogActActionConfig = field(default_factory=RoboTwin2CogActActionConfig)


@dataclass
class RoboTwin2CogActModelConfig(_CogACTModelConfig):
    """
    Model configuration class - controls model-related parameters

    Modification instructions:
    - model_name_or_path: Pre-trained model path or HuggingFace model name
    """
    # Pre-trained model path
    model_name_or_path: str = field(default='./checkpoints/Dexbotic-CogACT-HArm')
    action_dim: int = field(default=16)


@dataclass
class RoboTwin2CogActInferenceConfig(_CogACTInferenceConfig):
    """
    Inference configuration class - controls inference service parameters

    Modification instructions:
    - model_name_or_path: Model path for inference (if empty, uses model from training config)
    - port: Inference service port number
    """
    # NOTE: Inference model path - modify to the path you want to evaluate
    model_name_or_path: str = field(default='./checkpoints/robotwin2/db-cogact/adjust_bottle')
    # NOTE: action norm stats file path - modify to the norm stats file path corresponding to the inference model
    norm_stats: str = field(default='./checkpoints/robotwin2/db-cogact/adjust_bottle/norm_stats.json')
    # Inference service port number - modify to an available port
    port: int = field(default=7891)


@dataclass
class CogActExp(_CogACTExp):
    """
    Main experiment class - integrates all configurations and executes training or inference

    Modification instructions:
    This class contains instances of all configurations, you can customize parameters by modifying the configuration classes above
    """
    # Model configuration - controls model-related parameters
    model_config: RoboTwin2CogActModelConfig = field(default_factory=RoboTwin2CogActModelConfig)
    # Trainer configuration - controls training process parameters
    trainer_config: RoboTwin2CogActTrainerConfig = field(default_factory=RoboTwin2CogActTrainerConfig)
    # Data configuration - controls dataset and data processing parameters
    data_config: RoboTwin2CogActDataConfig = field(default_factory=RoboTwin2CogActDataConfig)
    # Inference configuration - controls inference service parameters
    inference_config: RoboTwin2CogActInferenceConfig = field(
        default_factory=RoboTwin2CogActInferenceConfig)
    
    def _auto_compute_norm_stats(self) -> None:
        if not self.data_config.auto_norm or self.data_config.action_config.statistic_mapping is not None:
            return
        norm_config = RoboTwin2CogActComputeNormActionConfig(
            delta=self.data_config.action_config.delta,
            norm_method=self.data_config.auto_norm_method)
        save_name = hashlib.md5(self.data_config.dataset_name.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name)
        norm_file_path = os.path.join(norm_config.norm_save_path, 'norm_stats.json')
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info('Auto-computing norm stats on rank0')
            norm_config.compute_norm_stats(self.data_config.dataset_name)
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                print(
                    f'Waiting for norm stats: {norm_file_path} to be computed on rank{self.local_rank}')
        self.data_config.action_config.statistic_mapping = norm_file_path


if __name__ == "__main__":
    """
    Main program entry point

    Usage:
    - Training mode: deepspeed playground/example_exp.py
    - Inference mode: python playground/example_exp.py --task inference

    Before running, modify the paths and parameters in each configuration class
    """
    args = parse_args()
    exp = CogActExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
