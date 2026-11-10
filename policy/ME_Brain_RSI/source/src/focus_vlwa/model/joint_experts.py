from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class JointExperts(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        world_model_config=None,
    ):
        if use_adarms is None:
            use_adarms = [False, False] if world_model_config is None else [False, False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.vision_language_model = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.action_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.action_expert.model.embed_tokens = None

        self.world_model_expert = None
        if world_model_config is not None:
            world_model_adarms = use_adarms[2] if len(use_adarms) > 2 else use_adarms[1]
            world_model_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=world_model_config.head_dim,
                hidden_size=world_model_config.width,
                intermediate_size=world_model_config.mlp_dim,
                num_attention_heads=world_model_config.num_heads,
                num_hidden_layers=world_model_config.depth,
                num_key_value_heads=world_model_config.num_kv_heads,
                vocab_size=257152,
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype="float32",
                use_adarms=world_model_adarms,
                adarms_cond_dim=world_model_config.width if world_model_adarms else None,
            )
            self.world_model_expert = GemmaForCausalLM(config=world_model_config_hf)
            self.world_model_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor, interpolate_pos_encoding: bool = False):
        if not interpolate_pos_encoding:
            return self.vision_language_model.model.get_image_features(image)
        # Native-res hist (e.g. 84²): interpolate 16×16 pos-embed to the patch grid.
        mdl = self.vision_language_model.model
        tower = getattr(mdl, "vision_tower", None) or getattr(self.vision_language_model, "vision_tower")
        proj = getattr(mdl, "multi_modal_projector", None) or getattr(
            self.vision_language_model, "multi_modal_projector"
        )
        out = tower(image, interpolate_pos_encoding=True)
        return proj(out.last_hidden_state)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.vision_language_model.language_model.embed_tokens(tokens)

    def expert_models(self):
        models = [self.vision_language_model.language_model, self.action_expert.model]
        if self.world_model_expert is not None:
            models.append(self.world_model_expert.model)
        return models

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None] * len(inputs_embeds)
        elif len(adarms_cond) < len(inputs_embeds):
            adarms_cond = list(adarms_cond) + [None] * (len(inputs_embeds) - len(adarms_cond))

        world_model_embedding = inputs_embeds[2] if len(inputs_embeds) > 2 else None
        if inputs_embeds[1] is None and world_model_embedding is None:
            prefix_output = self.vision_language_model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            outs = [prefix_output] + [None] * (len(inputs_embeds) - 1)
            return outs, prefix_past_key_values
        if inputs_embeds[0] is None and world_model_embedding is None:
            suffix_output = self.action_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            outs = [None, suffix_output] + [None] * (len(inputs_embeds) - 2)
            return outs, None

        else:
            models = self.expert_models()[: len(inputs_embeds)]
            num_layers = self.vision_language_model.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.action_expert.model, "gradient_checkpointing")
                and self.action_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = self.expert_models()[: len(inputs_embeds)]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    if hidden_states is None:
                        query_states.append(None)
                        key_states.append(None)
                        value_states.append(None)
                        gates.append(None)
                        continue
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention (skip None experts)
                query_states = torch.cat([q for q in query_states if q is not None], dim=2)
                key_states = torch.cat([k for k in key_states if k is not None], dim=2)
                value_states = torch.cat([v for v in value_states if v is not None], dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.vision_language_model.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.vision_language_model.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.vision_language_model.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.vision_language_model.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    if hidden_states is None:
                        outputs_embeds.append(None)
                        continue
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    if hidden_states is None:
                        outputs_embeds.append(None)
                        continue
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_past_key_values = None
            return outputs_embeds, prefix_past_key_values
