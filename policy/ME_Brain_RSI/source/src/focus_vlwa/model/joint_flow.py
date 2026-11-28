"""JointFlowMatching operations for Focus-VLWA."""

import torch

from focus_vlwa.model.attention import make_att_2d_masks


class JointFlowMatching:
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, *, event_noise=None, world_noise=None):
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors).

        With WM expert: run VLM + action + WM each Euler step.
        Returns ``{actions, event_action, world_state}``. Action comes from the action expert;
        WM denoises event_action + world_state only. ``world_state`` is 270×32-D
        (10 hops × head 3×3 + Lw 3×3 + Rw 3×3).
        Without WM: returns the action tensor only.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        bsize = observation.state.shape[0]
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        if noise is None:
            noise = self.sample_noise(actions_shape, device)
        elif tuple(noise.shape) != actions_shape:
            raise ValueError(f"Action noise must have shape {actions_shape}")

        images, img_masks, lang_tokens, lang_masks, state, proc_obs = self._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, proc_obs
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        if not self.use_world_model:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.joint_experts.vision_language_model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = self.joint_experts.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                )
                x_t = x_t + dt * v_t
                time += dt
            return x_t

        # Joint VLM + action + WM. Action expert owns the 50-step chunk;
        # WM owns event + world_state. They share a bidirectional attention block.
        x_act = noise
        event_shape = (bsize, self.world_model_event_tokens, self.world_model_dim)
        world_shape = (bsize, self.world_model_horizon, self.world_model_dim)
        x_ev = self.sample_noise(event_shape, device) if event_noise is None else event_noise.clone()
        noisy_world_state = self.sample_noise(world_shape, device) if world_noise is None else world_noise.clone()
        if tuple(x_ev.shape) != event_shape or tuple(noisy_world_state.shape) != world_shape:
            raise ValueError("Event or world noise has an invalid shape")
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            noisy_world_model = torch.cat([x_ev, noisy_world_state], dim=1)
            v_act, world_model_velocity = self.denoise_step_with_world_model(
                state,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                x_act,
                noisy_world_model,
                expanded_time,
            )
            x_act = x_act + dt * v_act
            noisy_world_model = noisy_world_model + dt * world_model_velocity
            x_ev = noisy_world_model[:, : self.world_model_event_tokens]
            noisy_world_state = noisy_world_model[:, self.world_model_event_tokens :]
            time += dt
        ev_out = x_ev[:, 0] if x_ev.ndim == 3 else x_ev
        self.last_event_action = ev_out
        self.last_world_state = noisy_world_state
        return {"actions": x_act, "event_action": ev_out, "world_state": noisy_world_state}

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.joint_experts.action_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.joint_experts.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_output_projection(suffix_out)

    def denoise_step_with_world_model(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        x_act,
        noisy_world_model,
        timestep,
    ):
        """One Euler step with VLM + action + WM. Returns (v_act, world_model_velocity)."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_act, timestep)
        world_model_embeddings, world_model_padding, world_model_attention = self.embed_world_model(noisy_world_model)
        if (
            self.joint_experts.vision_language_model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            world_model_embeddings = world_model_embeddings.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks, world_model_padding], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks, world_model_attention], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        outs, _ = self.joint_experts.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs, world_model_embeddings],
            use_cache=False,
            adarms_cond=[None, adarms_cond, adarms_cond],
        )
        suffix_out = outs[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        v_act = self.action_output_projection(suffix_out)
        world_model_velocity = self.world_model_output_projection(outs[2].to(dtype=torch.float32))
        return v_act, world_model_velocity
