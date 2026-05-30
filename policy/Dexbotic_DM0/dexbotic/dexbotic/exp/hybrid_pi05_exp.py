import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import megfile
import numpy as np
import torch
import transformers
from easydict import EasyDict
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BaseImageProcessor

import dexbotic.data.utils.normalize as normalize
from dexbotic.data.dataset.dex_dataset import DexDataset
from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddAction,
    AddTrajectory,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import (
    AddActionFlag,
    AddStateFlag,
    AddTextFlag,
    Pipeline,
    ToDict,
    ToList,
    ToNumpy,
    ToTensor,
)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
from dexbotic.exp.base_exp import ActionConfig, BaseExp, ModelConfig
from dexbotic.exp.pi0_exp import (
    Pi0ComputeNormActionConfig,
    Pi0DataConfig,
    Pi0InferenceConfig,
    Pi0OptimizerConfig,
    Pi0TokenizerConfig,
    Pi0TrainerConfig,
)
from dexbotic.model.pi05.hybrid_pi05_arch import HybridPi05ForCausalLM
from dexbotic.tokenization.process import Pi05Tokenization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class Pi05ModelConfig(ModelConfig):
    model_name_or_path: str = field(default="./checkpoints/Dexbotic-PI05")

    def build_model(self) -> HybridPi05ForCausalLM:
        model = HybridPi05ForCausalLM.from_pretrained(self.model_name_or_path)
        model.model.config.chunk_size = 50
        return model


class Pi05TrainerConfig(Pi0TrainerConfig):
    model_max_length: int = field(default=200)


@dataclass
class Pi05ComputeNormActionConfig(Pi0ComputeNormActionConfig):
    max_compute_iter: int = field(default=2000)

    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(predict_length=1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ToList(select_frame=True),
                AddStateFlag(empty_state_value=np.zeros(7)),
                AddActionFlag(empty_action_value=np.zeros((50, 7))),
                AddTextFlag(),
            ]
        )

        return action_config

    def _process_one_dataset(self, dataset_name, dataset):
        batch_size = 128
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=64
        )

        norm_keys = ["state", "action"]
        stats = {key: normalize.RunningStats() for key in norm_keys}
        compute_iter = min(self.max_compute_iter, len(dataloader))
        for batch_idx, batch in tqdm(
            enumerate(dataloader), desc="Computing norm stats", total=compute_iter + 1
        ):
            if batch_idx > compute_iter:
                break
            for key in norm_keys:
                values = batch[key].numpy()
                stats[key].update(values.reshape(-1, values.shape[-1]))
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

        save_path = os.path.join(self.norm_save_path, dataset_name)
        logger.info(f"Saving norm stats to {save_path}")
        normalize.save(save_path, norm_stats)

        return os.path.join(save_path, "norm_stats.json")

    def _merge_norm_stats(
        self, norm_files, per_task_norm=False, norm_keys=["action", "state"]
    ):
        norm_stats = {
            "default": {"min": -1, "max": 1},
        }
        for norm_key in norm_keys:
            min_list = []
            max_list = []
            mean_list = []
            std_list = []
            for dataset_name, (norm_file, dataset_path) in norm_files.items():
                with open(norm_file, "r") as f:
                    stats = json.load(f)["norm_stats"][norm_key]
                if per_task_norm:
                    norm_stats[dataset_path] = {
                        "default": {
                            "min": stats["q01"],
                            "max": stats["q99"],
                            "mean": stats["mean"],
                            "std": stats["std"],
                        }
                    }
                min_list.append(stats["q01"])
                max_list.append(stats["q99"])
                mean_list.append(stats["mean"])
                std_list.append(stats["std"])
            min_list = np.array(min_list)
            max_list = np.array(max_list)
            mean_list = np.array(mean_list)
            std_list = np.array(std_list)
            min_list = min_list.reshape(-1, min_list.shape[-1]).min(axis=0).tolist()
            max_list = max_list.reshape(-1, max_list.shape[-1]).max(axis=0).tolist()
            mean_list = mean_list.reshape(-1, mean_list.shape[-1]).mean(axis=0).tolist()
            std_list = std_list.reshape(-1, std_list.shape[-1]).mean(axis=0).tolist()
            norm_stats[norm_key] = {
                "min": min_list,
                "max": max_list,
                "mean": mean_list,
                "std": std_list,
            }

        with open(os.path.join(self.norm_save_path, "norm_stats.json"), "w") as f:
            json.dump({"norm_stats": norm_stats}, f, indent=2)


@dataclass
class Pi05ActionConfig(ActionConfig):
    trajectory_length: int = field(default=50)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(predict_length=1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ActionNorm(statistic_mapping=statistic_mapping, strict=False, use_quantiles=True),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                LoadMultiModal(return_masks=True),
                ToList(select_frame=True),
                AddStateFlag(
                    empty_state_value=np.zeros(
                        32,
                    )
                ),
                AddActionFlag(
                    empty_action_value=np.zeros((self.trajectory_length, 32))
                ),
                AddTextFlag(),
            ]
        )

        return action_config


@dataclass
class Pi05TokenizerConfig(Pi0TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class Pi05DataConfig(Pi0DataConfig):
    num_images: int = field(default=3)
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids",
            "labels",
            "action",
            "image",
            "state",
            "image_masks",
            "has_action",
            "has_text",
        ]
    )
    aug_policy: str | list[str] = field(
        default_factory=lambda: ["pi0", "color", "identity"]
    )
    action_config: Pi05ActionConfig = field(default_factory=Pi05ActionConfig)
    discrete_state_input: bool = field(default=True)

    def _build_dataset(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        image_processor: BaseImageProcessor,
    ) -> DexDataset:
        # FIXME: DO NOT USE EASYDICT IN NEXT VERSION
        data_args = EasyDict(
            {
                "dataset_name": self.dataset_name,
                "num_images": self.num_images,
                "data_keys": self.data_keys,
                "images_keys": self.images_keys,
                "aug_policy": self.aug_policy,
                "image_aspect_ratio": self.image_aspect_ratio,
                "image_processor": image_processor,
                "chat_template": chat_template,
                "image_pad_mode": self.image_pad_mode,
                "discrete_state_input": self.discrete_state_input,
            }
        )
        action_process_func = self.action_config.build_action_process_func()
        tokenization_func = Pi05Tokenization(tokenizer, data_args)
        dataset = DexDataset(
            data_args=data_args,
            tokenization_func=tokenization_func,
            action_process_func=action_process_func,
        )
        return dataset


@dataclass
class Pi05InferenceConfig(Pi0InferenceConfig):
    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = HybridPi05ForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=False
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.tokenization_func = Pi05Tokenization(self.tokenizer)
        logger.info("Model loaded successfully")

        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=True
                ),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=True
                ),
                AbsoluteAction(),
            ]
        )


@dataclass
class Pi05Exp(BaseExp):
    model_config: Pi05ModelConfig = field(default_factory=Pi05ModelConfig)
    optimizer_config: Pi0OptimizerConfig = field(default_factory=Pi0OptimizerConfig)
    trainer_config: Pi05TrainerConfig = field(default_factory=Pi05TrainerConfig)
    data_config: Pi05DataConfig = field(default_factory=Pi05DataConfig)
    tokenizer_config: Pi05TokenizerConfig = field(default_factory=Pi05TokenizerConfig)
    inference_config: Pi05InferenceConfig = field(default_factory=Pi05InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = Pi05ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)

    def _auto_compute_norm_stats(self) -> None:
        if (
            not self.data_config.auto_norm
            or self.data_config.action_config.statistic_mapping is not None
        ):
            return
        if self.local_rank == 0:
            print(
                f"Action config before auto compute norm: {self.data_config.action_config}"
            )
        _action_config = self.data_config.action_config
        norm_config = Pi05ComputeNormActionConfig()
        save_name = hashlib.md5(self.data_config.dataset_name.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name
        )
        norm_file_path = os.path.join(norm_config.norm_save_path, "norm_stats.json")
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info("Auto-computing norm stats on rank0")
            self.compute_norm_stats()
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                print(
                    f"Waiting for norm stats: {norm_file_path} to be computed on rank{self.local_rank}"
                )
        _action_config.statistic_mapping = norm_file_path
        self.data_config.action_config = _action_config
        if self.local_rank == 0:
            print(
                f"Action config after auto compute norm: {self.data_config.action_config}"
            )


if __name__ == "__main__":
    args = parse_args()
    exp = Pi05Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
