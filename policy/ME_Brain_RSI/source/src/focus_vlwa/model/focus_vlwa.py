import logging
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from focus_vlwa.configs.model import get_expert_config
from focus_vlwa.model import preprocessing as _preprocessing
from focus_vlwa.model.attention import make_att_2d_masks
from focus_vlwa.model.joint_experts import JointExperts
from focus_vlwa.model.joint_flow import JointFlowMatching
from focus_vlwa.model.prefix_encoder import VisualPrefixEncoder
from focus_vlwa.model.world_model import WorldModelExpert


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


class FocusVLWA(VisualPrefixEncoder, WorldModelExpert, JointFlowMatching, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_discrete_state = config.use_discrete_state

        vision_language_config = get_expert_config(config.vision_language_variant)
        action_expert_config = get_expert_config(config.action_expert_variant)
        self.use_world_model = bool(config.use_world_model)
        self.compute_world_model_supervision = True
        self.world_model_horizon = int(config.world_model_horizon)
        self.world_model_dim = int(config.world_model_dim)
        self.world_model_loss_weight = float(config.world_model_loss_weight)
        self.world_model_action_loss_weight = float(config.world_model_action_loss_weight)
        self.world_model_event_loss_weight = float(config.world_model_event_loss_weight)
        self.world_model_state_loss_weight = float(config.world_model_state_loss_weight)
        self.event_cell_loss_weight = 0.0
        # WM tokens = [1 event-action | 270 world_state × 32-D] (no 50-step action copy)
        self.world_model_event_tokens = 1
        self.world_model_total_horizon = self.world_model_event_tokens + self.world_model_horizon

        world_model_config = action_expert_config if self.use_world_model else None
        if self.use_discrete_state:
            use_adarms = [False, True, True] if self.use_world_model else [False, True]
        else:
            use_adarms = [False, False, False] if self.use_world_model else [False, False]

        self.joint_experts = JointExperts(
            vision_language_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=config.dtype,
            world_model_config=world_model_config,
        )

        self.action_input_projection = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_output_projection = nn.Linear(action_expert_config.width, config.action_dim)
        if self.use_world_model:
            # Same Linear as action expert; filled by copy_action_expert_into_world_model().
            self.world_model_input_projection = nn.Linear(self.world_model_dim, action_expert_config.width)
            self.world_model_output_projection = nn.Linear(action_expert_config.width, self.world_model_dim)

        if self.use_discrete_state:
            self.time_mlp_input = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_output = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_projection = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_input = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_output = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = (
            "The Focus-VLWA transformers patch is not installed. "
            "Run `focus-vlwa install-transformers-patch` in the active environment."
        )
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.joint_experts.vision_language_model.language_model.gradient_checkpointing = True
        self.joint_experts.vision_language_model.vision_tower.gradient_checkpointing = True
        self.joint_experts.action_expert.model.gradient_checkpointing = True
        if self.joint_experts.world_model_expert is not None:
            self.joint_experts.world_model_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for FocusVLWA model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.joint_experts.vision_language_model.language_model.gradient_checkpointing = False
        self.joint_experts.vision_language_model.vision_tower.gradient_checkpointing = False
        self.joint_experts.action_expert.model.gradient_checkpointing = False
        if self.joint_experts.world_model_expert is not None:
            self.joint_experts.world_model_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for FocusVLWA model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            observation,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.use_discrete_state:
            if self.state_projection.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_projection_func(state):
                return self.state_projection(state)

            state_emb = self._apply_checkpoint(state_projection_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_input_projection.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_input_projection(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.use_discrete_state:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_input(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_output(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_input(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_output(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Block start: VLM cannot attend to action. Action tokens are bidirectional
        # with each other; WM joins this same block (see embed_world_model).
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, proc_obs = self._preprocess_observation(
            observation, train=True
        )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, proc_obs
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        world_model_embeddings = world_model_padding = world_model_attention = None
        world_model_target_velocity = None
        world_model_mask = None
        compute_supervision = self.use_world_model and self.compute_world_model_supervision
        if self.use_world_model:
            world_state = getattr(observation, "world_state", None)
            if world_state is None:
                world_state = torch.zeros(
                    actions.shape[0],
                    self.world_model_horizon,
                    self.world_model_dim,
                    device=actions.device,
                    dtype=torch.float32,
                )
                world_state_mask = torch.zeros(
                    actions.shape[0],
                    self.world_model_horizon,
                    device=actions.device,
                    dtype=torch.float32,
                )
            else:
                world_state = world_state.to(device=actions.device, dtype=torch.float32)
                if world_state.shape[1] != self.world_model_horizon:
                    world_state = _match_horizon(world_state, self.world_model_horizon)
                world_state_mask = getattr(observation, "world_state_mask", None)
                if world_state_mask is None:
                    world_state_mask = torch.ones(
                        world_state.shape[0],
                        world_state.shape[1],
                        device=world_state.device,
                        dtype=torch.float32,
                    )
                else:
                    world_state_mask = world_state_mask.to(device=world_state.device, dtype=torch.float32)
                    if world_state_mask.shape[1] != self.world_model_horizon:
                        world_state_mask = _match_horizon(world_state_mask, self.world_model_horizon)
            ev = getattr(observation, "event_action", None)
            if ev is None:
                ev = torch.zeros(actions.shape[0], self.world_model_dim, device=actions.device, dtype=torch.float32)
                ev_mask = torch.zeros(actions.shape[0], 1, device=actions.device, dtype=torch.float32)
            else:
                ev = ev.to(device=actions.device, dtype=torch.float32)
                if ev.ndim == 2:
                    ev = ev.unsqueeze(1)
                elif ev.ndim == 3 and ev.shape[1] != 1:
                    ev = ev[:, :1]
                if ev.shape[-1] != self.world_model_dim:
                    ev = _match_last_dim(ev, self.world_model_dim)
                ev_m = getattr(observation, "event_action_mask", None)
                if ev_m is None:
                    ev_mask = torch.ones(ev.shape[0], 1, device=ev.device, dtype=torch.float32)
                else:
                    ev_mask = ev_m.to(device=ev.device, dtype=torch.float32).reshape(ev.shape[0], 1)
            noise_ev = self.sample_noise(ev.shape, ev.device)
            x_ev = time_expanded * noise_ev + (1 - time_expanded) * ev
            u_ev = noise_ev - ev

            noise_world_state = self.sample_noise(world_state.shape, world_state.device)
            noisy_world_state = time_expanded * noise_world_state + (1 - time_expanded) * world_state
            world_state_target_velocity = noise_world_state - world_state
            noisy_world_model = torch.cat([x_ev, noisy_world_state], dim=1)
            if compute_supervision:
                world_model_target_velocity = torch.cat([u_ev, world_state_target_velocity], dim=1)
                world_model_mask = torch.cat([ev_mask, world_state_mask], dim=1)
            world_model_embeddings, world_model_padding, world_model_attention = self.embed_world_model(
                noisy_world_model
            )
        if (
            self.joint_experts.vision_language_model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            if world_model_embeddings is not None:
                world_model_embeddings = world_model_embeddings.to(dtype=torch.bfloat16)

        if self.use_world_model:
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks, world_model_padding], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks, world_model_attention], dim=1)
        else:
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, world_model_embeddings, att_2d_masks_4d, position_ids, adarms_cond):
            embeds = [prefix_embs, suffix_embs]
            conds = [None, adarms_cond]
            if self.use_world_model:
                embeds.append(world_model_embeddings)
                # Same FM timestep as action (adaRMS), copied action-expert weights expect it.
                conds.append(adarms_cond)
            outs, _ = self.joint_experts.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=embeds,
                use_cache=False,
                adarms_cond=conds,
            )
            return outs[1], (outs[2] if compute_supervision else None)

        suffix_out, world_model_output = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, world_model_embeddings, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_output_projection_func(suffix_out):
            return self.action_output_projection(suffix_out)

        v_t = self._apply_checkpoint(action_output_projection_func, suffix_out)

        per = F.mse_loss(u_t, v_t, reduction="none")
        mask = getattr(observation, "action_mask", None)
        if mask is not None:
            m = mask.to(dtype=per.dtype, device=per.device)
            if m.ndim == 2:
                m = m.unsqueeze(-1)
            per = per * m
        if not self.use_world_model:
            return per

        if not compute_supervision:
            return {"action": per}

        def world_model_output_projection_func(world_model_output):
            return self.world_model_output_projection(world_model_output)

        world_model_velocity = self._apply_checkpoint(
            world_model_output_projection_func,
            world_model_output.to(dtype=torch.float32),
        )
        world_model_losses = F.mse_loss(world_model_target_velocity, world_model_velocity, reduction="none")
        if world_model_mask is not None:
            sm = world_model_mask.to(dtype=world_model_losses.dtype, device=world_model_losses.device)
            if sm.ndim == 2:
                sm = sm.unsqueeze(-1)
            world_model_losses = world_model_losses * sm
        ev_end = self.world_model_event_tokens
        return {
            "action": per,
            "world_model": world_model_losses,
            "world_model_event": world_model_losses[:, :ev_end],
            "world_model_state": world_model_losses[:, ev_end:],
        }


def _match_horizon(x: torch.Tensor, horizon: int) -> torch.Tensor:
    """Pad or crop sequence dim (axis=1) to `horizon`."""
    cur = x.shape[1]
    if cur == horizon:
        return x
    if cur > horizon:
        return x[:, :horizon]
    pad = horizon - cur
    if x.ndim == 2:
        return F.pad(x, (0, pad))
    return F.pad(x, (0, 0, 0, pad))


def _match_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    cur = x.shape[-1]
    if cur == dim:
        return x
    if cur > dim:
        return x[..., :dim]
    return F.pad(x, (0, dim - cur))
