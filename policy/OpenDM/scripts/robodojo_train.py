"""Full RoboDojo memory SFT; all extensions stay outside the upstream tree."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
from importlib.metadata import distributions

import albumentations as A
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint

from opendm.constants.robot import ActionMode
from opendm.data.augmentations import NoAugmentationPipeline
from opendm.data.collator import TrainingCollator
from opendm.data.dataset import JsonlDataset
from opendm.data.transforms import (
    ChatTokenization,
    LoadHistory,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
)
from opendm.dataset.register import register_dataset
from opendm.exp.dm05_exp import DM05DataConfig as BaseDataConfig
from opendm.exp.dm05_exp import DM05Exp as BaseExp
from opendm.exp.dm05_exp import DM05ModelConfig as BaseModelConfig
from opendm.exp.dm05_exp import DM05OptimizerConfig as BaseOptimizerConfig
from opendm.exp.dm05_exp import DM05TrainerConfig as BaseTrainerConfig
from XPolicyLab.policy.OpenDM.model import build_state_desc, check_latent_action_width, resolve_embodiment

from .recipe import (
    BASE_LR,
    BASE_REVISION,
    CheckpointPromptProcessor,
    IMAGE_KEYS,
    IMAGE_PROMPTS,
    POLICY_REVISION,
    RELEASE_STEPS,
    UPSTREAM_COMMIT,
    learning_rate,
    scheduler_kwargs,
)
from .robodojo_dm05_history import DM05InferenceConfig


class HistoryPixelTransform:
    """Apply color jitter to current views and only pad/resize history frames."""

    def __init__(self):
        resize = NoAugmentationPipeline()
        self.current = PixelTransform(
            A.Compose(
                [
                    A.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3, hue=0.0, p=0.5),
                    *resize.steps,
                ]
            )
        )
        self.history = PixelTransform(NoAugmentationPipeline())

    def __call__(self, data):
        history = data.pop("history_images")
        data = self.current(data)
        data["history_images"] = self.history({"images": history})["images"]
        return data


@dataclass
class DM05ModelConfig(BaseModelConfig):
    chunk_size: int = 50
    # Shared latent action width (``DM05Config.action_dim``). 32 is the released
    # DM0.5-Mem head and holds the 14-D joint vector; the 58-D WUJI hybrid trains with 64.
    latent_action_dim: int = 32
    precision_policy: str = "fp32_mixed"
    bf16: bool = True
    llm_attn_implementation: str = "flex_attention"
    vision_attn_implementation: str = "flash_attention_2"
    action_attn_implementation: str = "sdpa"

    def _config_overrides(self) -> dict:
        return {**super()._config_overrides(), "action_dim": int(self.latent_action_dim)}

    def _load_base_checkpoint_model(self):
        from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration

        config = DM05Config.from_pretrained(self.model_name_or_path)
        base_width = int(config.action_dim)
        for attr, value in self._config_overrides().items():
            setattr(config, attr, value)
        # A wider head than the base checkpoint re-initializes action_in_proj /
        # action_out_proj; every other weight loads unchanged.
        return DM05ForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            config=config,
            dtype=self._torch_dtype(),
            ignore_mismatched_sizes=base_width != int(self.latent_action_dim),
        )


@dataclass
class DM05OptimizerConfig(BaseOptimizerConfig):
    optim: str = "muon_adamw"
    base_lr: float = BASE_LR
    warmup_steps: int = 1000
    weight_decay: float = 1e-10
    adam_beta2: float = 0.95


@dataclass
class DM05TrainerConfig(BaseTrainerConfig):
    num_train_steps: int = RELEASE_STEPS
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 16  # Launcher recalculates for world size.
    save_steps: int = 10000
    save_total_limit: int = 100
    save_only_model: bool = False
    model_max_length: int = 1536
    tf32: bool = False
    seed: int = 0
    lr_scheduler_kwargs: dict = field(default_factory=scheduler_kwargs)


@dataclass
class DM05DataConfig(BaseDataConfig):
    dataset_name: str = "xpl_robodojo_sim"
    data_root: str = ""
    action_mode: ActionMode = ActionMode.ABSOLUTE
    is_history: bool = True
    robot_type: str = "Dual ARX5"
    # Set by the launcher from the model config; PadAction pads (never truncates) to it.
    latent_action_dim: int = 32
    compute_norm_stats_if_missing: bool = False

    def manifest(self):
        return json.loads((Path(self.data_root) / "dataset.json").read_text())

    def embodiment(self):
        manifest = self.manifest()
        return resolve_embodiment(
            manifest["env_cfg_type"], manifest.get("action_type", "joint"), manifest["robot_action_dim_info"]
        )

    def _dataset_info(self):
        root = Path(self.data_root)
        manifest = self.manifest()
        embodiment = self.embodiment()
        if manifest["action_dim"] != embodiment["action_dim"]:
            raise ValueError(
                f"dataset.json action_dim={manifest['action_dim']} but the "
                f"{embodiment['name']} contract is {embodiment['action_dim']}-D"
            )
        register_dataset(
            {
                self.dataset_name: {
                    "jsonl_dir": str(root / "jsonl"),
                    "image_dir": manifest["image_dir"],
                    "image_keys": IMAGE_KEYS,
                    "image_prompts": IMAGE_PROMPTS,
                    "robot_type": self.robot_type,
                    "state_desc": build_state_desc(embodiment),
                    "fps": manifest["fps"],
                    "speed": "0.5",
                }
            }
        )
        return super()._dataset_info()

    def build_dataset(self, processor, action_horizon, tokenizer_max_length=1536):
        info = self._dataset_info()
        pipeline = Pipeline(
            [
                self._action_transform(action_horizon),
                LoadImages(image_keys=IMAGE_KEYS, image_dir=info["image_dir"]),
                LoadHistory(
                    image_key="images_1",
                    image_dir=info["image_dir"],
                    max_history_images=20,
                    uniform_fps=1.0,
                ),
                HistoryPixelTransform(),
                Normalize(
                    norm_stats_path=str(self.norm_stats_path(action_horizon)),
                    norm_keys=["state", "action"],
                    use_quantiles=True,
                    clip_to_bounds=False,
                ),
                ChatTokenization(
                    processor=CheckpointPromptProcessor(processor),
                    n_bins=self.n_bins,
                    max_length=tokenizer_max_length,
                    image_prompts=IMAGE_PROMPTS,
                    add_state=True,
                    is_history=True,
                    max_history_images=20,
                ),
                # The model's shared latent action width; the robot width comes from
                # metadata and train_release() refuses a latent narrower than it.
                PadAction(int(self.latent_action_dim)),
            ]
        )
        dataset = JsonlDataset(
            jsonl_dir=info["jsonl_dir"],
            transforms=pipeline,
            dataset_name=self.dataset_name,
            dataset_meta=self._dataset_meta(info),
        )
        return dataset, TrainingCollator(processor.tokenizer.pad_token_id, tokenizer_max_length)


def export_assets(directory, *, processor, norm_stats_path, config, global_step):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(directory)
    shutil.copyfile(norm_stats_path, directory / "norm_stats.json")
    (directory / "training_config.json").write_text(
        json.dumps(
            {**config, "global_step": global_step},
            indent=2,
            default=lambda v: v.value if isinstance(v, Enum) else str(v),
        )
        + "\n"
    )
    data_root = Path(config["experiment"]["data_config"]["data_root"])
    shutil.copyfile(data_root / "episodes.jsonl", directory / "training_episodes.jsonl")
    shutil.copyfile(data_root / "dataset.json", directory / "training_dataset.json")


class ReleaseArtifacts(TrainerCallback):
    def __init__(self, exp, config, stop_after=None):
        self.exp, self.config, self.stop_after = exp, config, stop_after

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_after is not None and state.global_step >= self.stop_after:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            export_assets(
                Path(args.output_dir) / f"checkpoint-{state.global_step}",
                processor=self.exp.processor,
                norm_stats_path=self.exp.data_config.norm_stats_path(
                    self.exp.model_config.chunk_size
                ),
                config=self.config,
                global_step=state.global_step,
            )


@dataclass
class DM05Exp(BaseExp):
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)

    def train_release(self, *, stop_after=None):
        set_seed(self.trainer_config.seed)
        root = Path(self.data_config.data_root)
        embodiment = self.data_config.embodiment()
        self.data_config.latent_action_dim = int(self.model_config.latent_action_dim)
        check_latent_action_width(embodiment["action_dim"], self.data_config.latent_action_dim)
        # Statistics: the released 14-D RoboDojo joint stats for arx_x5 (never the EEF
        # pretraining stats); for wuji58, or on request, q01/q99 computed from this dataset.
        stats = self.data_config.norm_stats_path(self.model_config.chunk_size)
        if not stats.is_file() and self.data_config.compute_norm_stats_if_missing:
            self.data_config.compute_norm_stats(
                self.model_config.chunk_size,
                num_workers=self.trainer_config.dataloader_num_workers,
            )
            stats = self.data_config.norm_stats_path(self.model_config.chunk_size)
        if not stats.is_file():
            raise FileNotFoundError(
                f"Download the RoboDojo normalization statistics first: {stats}"
            )
        config = {
            "experiment": asdict(self),
            "upstream_commit": UPSTREAM_COMMIT,
            "reference_base_revision": BASE_REVISION,
            "normalization_revision": POLICY_REVISION,
            "base_config_sha256": hashlib.sha256(
                (Path(self.model_config.model_name_or_path) / "config.json").read_bytes()
            ).hexdigest(),
            "dataset": json.loads((root / "dataset.json").read_text()),
            "episodes_sha256": hashlib.sha256((root / "episodes.jsonl").read_bytes()).hexdigest(),
            "norm_stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
            "world_size": int(os.environ["WORLD_SIZE"]),
            "global_batch_size": int(os.environ["WORLD_SIZE"])
            * self.trainer_config.per_device_train_batch_size
            * self.trainer_config.gradient_accumulation_steps,
            "schedule_steps": self.trainer_config.num_train_steps,
            "final_learning_rate": learning_rate(
                self.trainer_config.num_train_steps, self.trainer_config.num_train_steps
            ),
            "packages": {d.metadata["Name"]: d.version for d in distributions()},
            "embodiment": embodiment["name"],
            "latent_action_dim": self.data_config.latent_action_dim,
            "data_recipe": {
                "history_slots": 20,
                "history_fps": 1,
                "history_tokens_per_slot": 16,
                "normalization": "q01/q99",
                "clip_to_bounds": False,
                "current_augmentation": {
                    "brightness": 0.3,
                    "contrast": 0.4,
                    "saturation": 0.3,
                    "hue": 0,
                    "probability": 0.5,
                },
                "history_augmentation": "none",
                "image_transform": "pad to square, resize 448x448",
            },
        }
        self._initialize_train()
        config["training_arguments"] = self.trainer.args.to_dict()
        self.trainer.add_callback(ReleaseArtifacts(self, config, stop_after))
        output = Path(self.trainer_config.output_dir)
        resume = get_last_checkpoint(str(output)) if output.exists() else None
        self.trainer.train(resume_from_checkpoint=resume)
        self.trainer.save_state()
        self.model.config.use_cache = True
        self.model.model.vlm.config.use_cache = True
        # Trainer.save_model gathers FSDP weights before writing an HF directory.
        self.trainer.save_model(str(output))
        self.trainer.accelerator.wait_for_everyone()
        if self.trainer.args.should_save:
            export_assets(
                output,
                processor=self.processor,
                norm_stats_path=stats,
                config=config,
                global_step=self.trainer.state.global_step,
            )
