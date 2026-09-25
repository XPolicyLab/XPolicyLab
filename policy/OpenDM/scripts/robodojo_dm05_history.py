"""Released RoboDojo generalist inference recipe (50 predicted / 25 executed)."""

from dataclasses import dataclass, field
import os

import torch

from opendm.constants.robot import ActionMode
from opendm.data.transforms import ChatTokenization
from opendm.exp.dm05_exp import DM05DataConfig as BaseDataConfig
from opendm.exp.dm05_exp import DM05Exp as BaseExp
from opendm.exp.dm05_exp import DM05InferenceConfig as BaseInferenceConfig
from opendm.exp.dm05_exp import DM05ModelConfig as BaseModelConfig
from XPolicyLab.policy.OpenDM.scripts.recipe import CheckpointPromptProcessor


@dataclass
class DM05ModelConfig(BaseModelConfig):
    precision_policy: str = "fp32_mixed"
    bf16: bool = False
    llm_attn_implementation: str = "sdpa"
    vision_attn_implementation: str = "sdpa"
    action_attn_implementation: str = "sdpa"

    def _load_base_checkpoint_model(self):
        # Casting after loading cannot recover FP32 values rounded by nested
        # Gemma configs that still advertise BF16 in the checkpoint metadata.
        from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration

        config = DM05Config.from_pretrained(self.model_name_or_path)
        for attr, value in self._config_overrides().items():
            setattr(config, attr, value)
        dtype = self._torch_dtype()
        config.dtype = dtype
        for nested in (
            config.vlm_config,
            config.vlm_config.text_config,
            config.vlm_config.vision_config,
            config.action_config,
        ):
            nested.dtype = dtype

        llm_attn = self.llm_attn_implementation
        vision_attn = self.vision_attn_implementation
        if llm_attn != "auto" and vision_attn != "auto":
            vlm_attn = {
                "": llm_attn,
                "text_config": llm_attn,
                "vision_config": vision_attn,
            }
            config.vlm_config._attn_implementation = vlm_attn
            config.vlm_config.attn_implementation = vlm_attn
        if llm_attn != "auto":
            config.vlm_config.text_config._attn_implementation = llm_attn
            config.vlm_config.text_config.attn_implementation = llm_attn
        if vision_attn != "auto":
            config.vlm_config.vision_config._attn_implementation = vision_attn
            config.vlm_config.vision_config.attn_implementation = vision_attn
        if self.action_attn_implementation != "auto":
            config.action_config._attn_implementation = self.action_attn_implementation
            config.action_config.attn_implementation = self.action_attn_implementation

        return DM05ForConditionalGeneration.from_pretrained(
            self.model_name_or_path, config=config, dtype=dtype,
        )


@dataclass
class DM05DataConfig(BaseDataConfig):
    action_mode: ActionMode = ActionMode.ABSOLUTE
    is_history: bool = True


@dataclass
class DM05InferenceConfig(BaseInferenceConfig):
    max_history_images: int = 20
    clip_to_bounds: bool = False
    diffusion_noise_seed: int | None = None

    def _initialize(self, model, **kwargs):
        device = (
            f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
        )
        model.to(torch.device(device))
        super()._initialize(model=model, **kwargs)
        for transform in self.input_transform.transforms:
            if isinstance(transform, ChatTokenization):
                transform.processor = CheckpointPromptProcessor(transform.processor)

    def _predict(self, data):
        if self.diffusion_noise_seed is None:
            return super()._predict(data)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.diffusion_noise_seed)
            return super()._predict(data)


@dataclass
class DM05Exp(BaseExp):
    use_lora: bool = False
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)
