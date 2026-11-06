from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.names import WSA_BASE
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_PREFIX

from .configuration_wsa_base import WSABaseConfig
from .modeling_wsa_base import WSABaseModel, WSABasePolicy, make_att_2d_masks


class WSABaseRTCModel(WSABaseModel):
    """WSABase model variant with runtime-only RTC support for inference."""

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        num_steps=None,
        decode_image=False,
        rtc_processor=None,
        inference_delay=None,
        prev_chunk_left_over=None,
        execution_horizon=None,
    ) -> tuple[Tensor, Tensor | None]:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        if self.uses_causal_attention():
            rtc_enabled = (
                rtc_processor is not None
                and getattr(getattr(rtc_processor, "rtc_config", None), "enabled", False)
            )
            if rtc_enabled:
                raise ValueError("WSABase RTC inference is not supported with attention_mask_mode='causal'.")
            return super().sample_actions(
                images,
                img_masks,
                pixel_values,
                image_grid_thw,
                lang_tokens,
                lang_masks,
                state,
                noise=noise,
                num_steps=num_steps,
                decode_image=decode_image,
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, prefix_pad_masks)

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        with self._temporary_attention_implementations(
            und_expert_impl="eager",
            gen_expert_impl="eager",
            act_expert_impl="eager",
        ):
            _, past_key_values = self.qwen3_vl_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None, None],
                use_cache=True,
            )
            max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

            middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
                images[:, :, :2], img_masks,
            )

            middle_len = middle_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, middle_len, prefix_len)
            middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
            middle_att_2d_masks = self.apply_view_aware_query_attention(middle_att_2d_masks, prefix_len=0)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)

            middle_position_ids = torch.arange(1, middle_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids) + max_prefix_position_ids

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

            (_, middle_out, _), past_key_values = self.qwen3_vl_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=middle_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, middle_embs, None],
                use_cache=True,
            )

            max_position_ids = middle_position_ids.max(dim=-1, keepdim=True).values
            curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)
            suffix_att_2d_masks_4d, suffix_position_ids = self._prepare_suffix_denoise_cache(
                curr_pad_masks,
                max_position_ids,
            )

            dt = -1.0 / num_steps
            dt = torch.tensor(dt, dtype=torch.float32, device=device)

            x_t = noise
            if prev_chunk_left_over is not None:
                prev_chunk_left_over = prev_chunk_left_over.to(device=device, dtype=x_t.dtype)
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)

                def denoise_step_partial_call(input_x_t, current_timestep=expanded_time):
                    return self.denoise_step(
                        state,
                        past_key_values,
                        suffix_att_2d_masks_4d,
                        suffix_position_ids,
                        input_x_t.to(dtype),
                        current_timestep.to(dtype),
                    )

                rtc_enabled = (
                    rtc_processor is not None
                    and getattr(getattr(rtc_processor, "rtc_config", None), "enabled", False)
                )
                if rtc_enabled:
                    v_t = rtc_processor.denoise_step(
                        x_t=x_t,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay,
                        time=time,
                        original_denoise_step_partial=denoise_step_partial_call,
                        execution_horizon=execution_horizon,
                    )
                else:
                    v_t = denoise_step_partial_call(x_t)
                x_t = x_t + dt * v_t
                time += dt

            if decode_image:
                def cosmos_out_func(middle_out):
                    return self.decode_cosmos(middle_out)
                middle_visual_out, _ = self.split_middle_tokens(middle_out)
                decode_dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
                pred_cosmos_features = self._apply_checkpoint(
                    cosmos_out_func,
                    middle_visual_out.to(dtype=decode_dtype),
                )
                pred_cosmos_features = pred_cosmos_features.squeeze(0)
                recon_images = self.cosmos.decode(pred_cosmos_features.squeeze(0))
            else:
                recon_images = None

            return x_t, recon_images


class WSABaseRTCPolicy(WSABasePolicy):
    """WSABase policy variant that enables RTC only for explicit runtime use."""

    config_class = WSABaseConfig
    name = WSA_BASE

    def __init__(self, config: WSABaseConfig):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.model = WSABaseRTCModel(config)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        decode_image=False,
        inference_delay=None,
        prev_chunk_left_over=None,
        rtc_processor=None,
        execution_horizon=None,
    ) -> tuple[Tensor, Tensor | None]:
        self.eval()

        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        state = self.prepare_state(batch)

        images, img_masks = self._preprocess_images(batch)

        actions, recon_images = self.model.sample_actions(
            images,
            img_masks,
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            state,
            decode_image=decode_image,
            rtc_processor=rtc_processor,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions, recon_images
