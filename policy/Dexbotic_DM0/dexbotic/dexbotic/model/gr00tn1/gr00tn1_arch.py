from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CONFIG_MAPPING, AutoConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from dexbotic.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from dexbotic.model.dexbotic_arch import (
    ActionOutputForCausalLM,
    CausalLMOutputDexbotic,
    DexboticConfig,
    DexboticForCausalLM,
    DexboticVLMModel,
)
from dexbotic.model.gr00tn1.action_model.builder import build_action_model
from dexbotic.model.modules.mm_vision.siglip.gr00t_siglip_encoder import (
    GR00TSiglipVisionTower,
)


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """


class GR00TN1Config(DexboticConfig):
    model_type = "dexbotic_gr00tn1"
    action_model_type: Optional[str] = None
    action_head_config: Dict[str, Any] = None
    hidden_size = 1536

    select_layer = 12

    state_horizon = 1
    max_state_dim = 64
    action_horizon = 16
    action_dim = 7
    max_action_dim = 32

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    chat_template: str = "qwen2-chat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        llm_config = kwargs.get("llm_config", None)
        if isinstance(llm_config, dict):
            self.llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
        elif isinstance(llm_config, str):
            self.llm_config = AutoConfig.from_pretrained(llm_config)


class GR00TN1Model(DexboticVLMModel):
    def __init__(self, config: GR00TN1Config):
        super().__init__(config)
        self.embodiment_tag = EmbodimentTag(config.embodiment_tag)
        self.embodiment_tag_mapping = {
            EmbodimentTag.GR1: 24,
            EmbodimentTag.NEW_EMBODIMENT: 31,
        }
        while len(self.llm.layers) > config.select_layer:
            self.llm.layers.pop(-1)

        self.mm_vision_tower.vision_tower.vision_model.head = torch.nn.Identity()
        self.linear = nn.Linear(config.llm_config.hidden_size, 1536)

        if config.action_model_type is not None:
            self.action_head = build_action_model(config)

    def _merge_llm(self):
        # merge llm config with self.config, only add missing keys
        llm_config_dict = {
            k: v
            for k, v in self.llm.config.__dict__.items()
            if not k.startswith("_") and not hasattr(self.config, k)
        }
        for key, value in llm_config_dict.items():
            setattr(self.config, key, value)

    def _build_mm_vision_module(self, config) -> nn.Module:
        if getattr(self, "mm_vision_tower", None) is not None:
            return self.mm_vision_tower
        if (
            config["vision_config"] is not None
            and config["processor_config"] is not None
        ):
            # FIXME: processor should be moved to top level config
            vision_config = config["vision_config"]
            if vision_config["model_type"] == "siglip_vision_model":
                vision_config = SiglipVisionConfig(**vision_config)
            else:
                raise ValueError(
                    "Unsupported model_type: {}".format(vision_config["model_type"])
                )
            self.mm_vision_tower = GR00TSiglipVisionTower(
                vision_config,
                processor_config=config["processor_config"],
                select_layer=None,
            )
        else:
            raise ValueError("processor_config and vision_config are required")
        self.config.mm_hidden_size = self.mm_vision_tower.hidden_size

        return self.mm_vision_tower

    def _build_mm_projector_module(self, config) -> nn.Module:
        self.mm_projector = nn.Sequential(
            nn.LayerNorm(self.mm_vision_tower.hidden_size * 4),
            nn.Linear(
                self.mm_vision_tower.hidden_size * 4, self.llm.config.hidden_size
            ),
            nn.GELU(),
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size),
        )
        return self.mm_projector

    def set_trainable_parameters(
        self,
        tune_llm: bool,
        tune_visual: bool,
        tune_projector: bool,
        tune_diffusion_model: bool,
    ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model

        if not tune_llm:
            self.llm.requires_grad_(False)
        if not tune_visual:
            self.mm_vision_tower.vision_tower.requires_grad_(False)
            self.mm_projector.requires_grad_(False)
        self.action_head.set_trainable_parameters(
            tune_projector=self.tune_projector,
            tune_diffusion_model=self.tune_diffusion_model,
        )

    def _extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        def encode_image(image: torch.Tensor) -> torch.Tensor:
            image_features = self.mm_vision_module(image).contiguous()
            h = w = int(image_features.shape[1] ** 0.5)
            image_features = image_features.reshape(image_features.shape[0], h, w, -1)
            image_features = self.pixel_shuffle(image_features)
            image_features = image_features.reshape(
                image_features.shape[0], -1, image_features.shape[-1]
            )

            image_features = self.mm_projector_module(image_features)
            return image_features

        if images.ndim == 5:
            # [B n_image, C, H, W] -> [B*n_image, C, H, W]
            concat_images = torch.cat([image for image in images], dim=0)
            concat_image_features = encode_image(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(
                concat_image_features, split_sizes, dim=0
            )  # {[n_image n_token C] * B}
            # {[n_image*n_token C] * B}
            image_features = [x for x in image_features]

            image_features = torch.stack(image_features, dim=0)
            image_features = image_features.flatten(0, 1)
        else:
            image_features = encode_image(images)

        image_features = image_features.to(self.device)
        return image_features

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _insert_multimodal_embeds_per_batch(
        self, image_features, cur_input_ids, cur_labels, cur_image_idx
    ):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            # 用来跳过填充的空图像
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.backbone.embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat(
                [cur_input_embeds_1, cur_image_features[0:0]], dim=0
            )
            cur_image_idx += 1

            return cur_input_embeds, cur_labels, cur_image_idx

        image_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        image_token_indices = (
            [-1] + image_positions + [cur_input_ids.shape[0]]
        )  # [-1, image_index, end]

        cur_input_ids_noim = []
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(
                cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
            cur_labels_noim.append(
                cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
        # [0 -> image_index] [image_index+1 -> end]

        split_sizes = [x.shape[0] for x in cur_labels_noim]

        cur_input_embeds = self.backbone.get_input_embeddings()(
            torch.cat(cur_input_ids_noim)
        )
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])

            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full(
                        (cur_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )
                cur_image_idx += 1

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        return cur_new_input_embeds, cur_new_labels, cur_image_idx

    def _prepare_action_head_input(
        self,
        actions: torch.Tensor,
        state: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ):
        actions, actions_mask, _ = self._prepare_action(actions)
        state, state_mask, _ = self._prepare_state(state)
        if dtype is not None:
            actions = actions.to(dtype=dtype)
            state = state.to(dtype=dtype)
        if len(state.shape) == 2:
            state = state.unsqueeze(0).repeat(actions.shape[0], 1, 1)
            state_mask = state_mask.unsqueeze(0).repeat(actions.shape[0], 1, 1)
        return BatchFeature(
            data={
                "embodiment_id": torch.tensor(
                    [self.embodiment_tag_mapping[self.embodiment_tag]]
                )
                .repeat(actions.shape[0])
                .to(actions.device),
                "action": actions.to(actions.device),
                "action_mask": actions_mask.to(actions.device),
                "state": state.to(actions.device),
                "state_mask": state_mask.to(actions.device),
            }
        )

    def _prepare_state(self, state: torch.Tensor):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if state is None:
            state = torch.zeros((self.config.state_horizon, self.config.max_state_dim))
            state_mask = torch.zeros(
                (self.config.state_horizon, self.config.max_state_dim), dtype=bool
            )
            n_state_tokens = self.config.state_horizon
            return state, state_mask, n_state_tokens

        assert (
            state.shape[1] == self.config.state_horizon
        ), f"{state.shape=}, {self.config.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.config.max_state_dim:
            state = state[:, : self.config.max_state_dim]
            n_state_dims = self.config.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = F.pad(
                state,
                (0, self.config.max_state_dim - n_state_dims, 0, 0, 0, 0),
                mode="constant",
                value=0,
            )

        # Create mask for real state dims
        state_mask = torch.zeros_like(state, dtype=torch.bool)
        state_mask[:, :, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[1]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, actions: torch.Tensor):
        """
        Pad to max_action_dim, return masks.
        """
        if actions is None:
            raise ValueError("Actions cannot be None")

        actions = actions.reshape(
            -1, self.config.action_horizon, self.config.action_dim
        )
        batch_size = actions.shape[0]
        n_action_tokens = actions.shape[1]  # T
        n_action_dims = actions.shape[2]

        assert (
            n_action_dims <= self.config.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.config.max_action_dim}."

        # Pad the channel dimension
        # actions = np.pad(actions, ((0, 0), (0, 0), (0, self.config.max_action_dim - n_action_dims)), "constant")
        actions = F.pad(
            actions,
            (0, self.config.max_action_dim - n_action_dims, 0, 0, 0, 0),
            mode="constant",
            value=0,
        )
        # Create mask: [B, T, max_action_dim]
        # actions_mask = np.zeros((batch_size, n_action_tokens, self.config.max_action_dim), dtype=bool)
        actions_mask = torch.zeros(
            (batch_size, n_action_tokens, self.config.max_action_dim),
            dtype=torch.bool,  # 对应 NumPy 的 dtype=bool
        )
        actions_mask[:, :, :n_action_dims] = True
        actions_mask = actions_mask.to(actions.device)

        return actions, actions_mask, n_action_tokens


class GR00TN1ForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = GR00TN1Config

    def _real_init(self, config: GR00TN1Config):
        self.model = GR00TN1Model(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.LongTensor] = None,
        state: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            cache_position,
        ) = self.model._prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            cache_position,
            images,
        )
        llm_embeddings = self.model.llm.forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        llm_embeddings = llm_embeddings.hidden_states[-1]
        last_hidden_state = llm_embeddings
        llm_embeddings = self.model.linear(llm_embeddings)

        backbone_outputs = {
            "backbone_features": llm_embeddings,
            "backbone_attention_mask": attention_mask,
        }

        backbone_outputs = BatchFeature(data=backbone_outputs)
        action_inputs = self.model._prepare_action_head_input(
            actions, state, dtype=llm_embeddings.dtype
        )
        action_head_outputs = self.model.action_head(backbone_outputs, action_inputs)

        loss = action_head_outputs["loss"]
        if not return_dict:
            return (
                (loss,) + last_hidden_state if loss is not None else last_hidden_state
            )
        return CausalLMOutputDexbotic(
            loss=loss,
            logits=last_hidden_state,
        )

    @torch.no_grad()
    def inference_action(self, input_ids, image_tensor, inference_args={}, **kwargs):
        action_norms = inference_args.get("action_norms")
        state = inference_args.get("state", None)

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            cache_position,
        ) = self.model._prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
            cache_position=None,
            images=image_tensor,
        )

        llm_outputs = self.model.llm.forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        llm_embeddings = self.model.linear(llm_outputs.hidden_states[-1])

        backbone_outputs = BatchFeature(
            data={
                "backbone_features": llm_embeddings,
                "backbone_attention_mask": attention_mask,
            }
        )

        # dummy actions for _prepare_action_head_input
        batch_size = inputs_embeds.size(0)
        dummy_actions = torch.zeros(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            device=llm_embeddings.device,
            dtype=llm_embeddings.dtype,
        )
        action_inputs = self.model._prepare_action_head_input(
            dummy_actions, state, dtype=llm_embeddings.dtype
        )

        action_outputs = self.model.action_head.get_action(
            backbone_outputs, action_inputs
        )
        actions = action_outputs["action_pred"]  # [B, T, D]

        actions = actions[:, :, : self.config.action_dim]

        if action_norms is not None:
            actions = self._denorm(
                actions[0].float().cpu().numpy(), action_norms
            ).tolist()
        else:
            actions = actions[0].float().cpu().numpy().tolist()
        return actions
