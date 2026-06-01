import logging
import math
import json
import time as time_module

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi_value.models.gemma as _gemma
from openpi_value.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi_value.models_pytorch.preprocessing_pytorch as _preprocessing
import copy
from openpi_value.models.model import Observation
import random



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


def get_1d_sincos_pos_embed_from_grid(pos: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    Generates 1D sinusoidal positional embeddings in PyTorch.

    Args:
        embed_dim: Output dimension (D) for each position. Must be even.
        pos: A list or tensor of positions (M,) to be encoded. 
             If passed as a numpy array, PyTorch will convert it to a tensor.

    Returns:
        A tensor of shape (M, D) containing the positional embeddings.
    """
    
    # 1. Input assertion and dimension setup
    assert embed_dim % 2 == 0, "Embedding dimension must be an even number."
    
    # Ensure pos is a tensor and flatten it
    if isinstance(pos, torch.Tensor):
        pos = pos.flatten()
    else:
        # Assuming input is convertible (e.g., numpy array or list)
        pos = torch.as_tensor(pos, dtype=torch.float32).flatten()
        
    # M = pos.shape[0]  # Number of positions
    D_half = embed_dim // 2 # D/2
    
    # 2. Calculate omega (frequencies)
    # The original implementation uses 10000 as the base constant
    
    # Calculate indices for D/2 dimensions: 0, 1, 2, ..., D/2 - 1
    # Use torch.float32 (standard)
    omega = torch.arange(D_half, dtype=torch.float32).to(pos)
    
    # Apply the division: i / (D/2)
    omega = omega / D_half
    
    # Apply the base power: 1 / 10000^(i / (D/2))
    # torch.pow is safer than ** for tensors, or 10000.0 ** omega
    omega = 1.0 / torch.pow(10000.0, omega)  # (D/2,)

    # 3. Outer product (M, D/2)
    # The einsum "m,d->md" is equivalent to multiplying (M, 1) by (1, D/2)
    # which uses broadcasting, or using torch.einsum directly.
    # out = torch.einsum("m,d->md", pos, omega)
    
    # Using broadcasting for better performance/readability in PyTorch
    # pos shape: (M, 1), omega shape: (1, D/2) -> out shape: (M, D/2)
    out = pos.unsqueeze(1) * omega.unsqueeze(0)  

    # 4. Calculate sine and cosine components
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    # 5. Concatenate to get final embedding (M, D)
    emb = torch.cat([emb_sin, emb_cos], dim=1) 
    
    return emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

        [[1 1 1 1 1 1]]: pure causal attention.

        [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
            themselves and the last 3 tokens have a causal attention. The first
            entry could also be a 1 without changing behaviour.

        [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
            block can attend all previous blocks and all tokens on the same block.

    Args:
        input_mask: bool[B, N] true if its part of the input, false if padding.
        mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
            it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# * alpha = 1 / len(episode)
class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.with_value_head = getattr(config, "with_value_head", False)
        self.loss_value_weight = getattr(config, "loss_value_weight", 0.0)
        
        self.loss_value_use_bce = getattr(config, "loss_value_use_bce", False)
        assert self.loss_value_use_bce == False, "BCE loss is currently not supported due to the presence of negative progress labels. Please set loss_value_use_bce to False."
        
        self.loss_action_weight = getattr(config, "loss_action_weight", 1.0)
        self.loss_value_td_weight = getattr(config, "loss_value_td_weight", 1.0)
        self.p_mask_ego_state = getattr(config, "p_mask_ego_state", 0.0)

        self.apply_shape_visual_aug = getattr(config, "apply_shape_visual_aug", False)
        self.apply_blur_visual_aug = getattr(config, "apply_blur_visual_aug", False)
        
        
        self.exist_negative_progress = getattr(config, "exist_negative_progress", False)
        
        self.p_mask_base = getattr(config, "p_mask_base", 0.0)
        
        self.state_noise_snr = getattr(config, "state_noise_snr", None)  # * SNR for state noise injection during training.
        
        self.fix_value_from_prefix = getattr(config, "fix_value_from_prefix", False)  # * Whether to fix the position of the value head in the suffix tokens.
        assert self.fix_value_from_prefix in [False, 'v1', 'v2'], f"Invalid fix_value_from_prefix: {self.fix_value_from_prefix}"
        # * one of [False, 'v1', 'v2']
        # * v1 still has bug
        
        self.freeze_vlm_backbone = getattr(config, "freeze_vlm_backbone", False)  # * Whether to freeze the VLM weights during training.
        
        # * TD args
        # * only in training mode.
        self.p_with_progress_loss = getattr(config, "p_with_progress_loss", 1.)
        # self.p_with_progress_loss = p_with_progress_loss and self.with_value_head

        TD_learning = getattr(config, "value_TD_learning", False)     # * apply TD learning for negative samples.
        self.TD_learning = TD_learning and self.with_value_head
        if self.TD_learning:
            self.TD_TAU = getattr(config, "value_TD_TAU", 0.005)
            self.gamma = getattr(config, "value_gamma", 0.99) # [ADDED] Discount factor
            # * should be similar to 1 / episode_length

            self.terminal_window = getattr(config, "value_terminal_window", 10)


            # * success episode reward goes from 1 to 0 gradually.
            self.failure_reward = getattr(config, "value_failure_reward", -1.0)
            self.success_reward = getattr(config, "value_success_reward", 1.0)
        

        # Value head is a 3-layer MLP that takes the last-layer representation of the suffix tokens and outputs a single value
        if self.with_value_head:
            mlp_layers = [
                nn.Linear(action_expert_config.width, action_expert_config.width),
                nn.SiLU(),  # Equivalent to swish activation
                nn.Linear(action_expert_config.width, action_expert_config.width),
                nn.SiLU(),  # Equivalent to swish activation
                nn.Linear(action_expert_config.width, 1),
            ]     
            if self.exist_negative_progress:
                # If using timestep difference mode, use tanh activation to bound output between [-1, 1]
                mlp_layers.append(nn.Tanh())
            elif not self.loss_value_use_bce:
                mlp_layers.append(nn.Sigmoid())

            self.value_head = nn.Sequential(*mlp_layers)
            

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        
        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

        
        if self.freeze_vlm_backbone:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
            logging.info("Froze VLM weights in PI0Pytorch model.")


    # * tested, the same
    def init_target_model(self):
        """Initializes the target model."""
        self.target_model = copy.deepcopy(self)
        self.target_model.eval()
        logging.info("Initialized Target Critic Network for TD Learning")
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.target_model.TD_learning = False
        self.target_model.target_model = None

        self.target_model.eval()
        logging.info("Initialized Target Critic Network for TD Learning")

    def update_target_network(self):
        """Updates the target network weights using Exponential Moving Average (EMA)."""
        assert self.TD_learning, "TD learning must be enabled to update the target network"
        if self.target_model is None:
            return

        with torch.no_grad():
            for param, target_param in zip(self.parameters(), self.target_model.parameters()):
                target_param.data.mul_(1 - self.TD_TAU)
                target_param.data.add_(param.data * self.TD_TAU)



    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

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


    # * TODO: random base image masking for value learning.
    def _preprocess_observation(self, observation, *, train=True, return_full_obs=False):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, 
                                                                    train=train, 
                                                                    return_full_obs=return_full_obs,
                                                                    apply_shape_visual_aug=self.apply_shape_visual_aug,
                                                                    apply_blur_visual_aug=self.apply_blur_visual_aug,
                                                                    p_mask_base=self.p_mask_base,
                                                                    state_noise_snr=self.state_noise_snr,
                                                                    )

        if return_full_obs:
            return (
                list(observation.images.values()),
                list(observation.image_masks.values()),
                observation.tokenized_prompt,
                observation.tokenized_prompt_mask,
                observation.state,
                observation, # Pass the whole observation object for value target calculation
            )

        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
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


    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            
            # --- Start of Modifications ---
            # Randomly mask the ego state during training
            if self.training and self.p_mask_ego_state > 0.0:
                mask = torch.bernoulli(torch.full((state.shape[0],), self.p_mask_ego_state, device=state.device)).bool()
                state[mask] = 0.0
            # --- End of Modifications ---
                
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)


            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None, return_loss_dict=False) -> tuple[Tensor, dict]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""


        if self.TD_learning:

            # * assign future images to future_obs, (other elements are not used for value learning input.)
            future_obs = {}
            for k, v in observation.__dict__.items():
                if k != "images":
                    future_obs[k] = v

            future_obs_images = {}
            for cam in ["base", "left_wrist", "right_wrist"]:
                src = f"{cam}_1_rgb"
                dst = f"{cam}_0_rgb"
                future_obs_images[dst] = observation.images[src]

            future_obs["image"] = future_obs_images
            future_obs["image_mask"] = future_obs.pop("image_masks")
            future_obs = Observation.from_dict(future_obs)
            observation.drop_images(["base_1_rgb", "left_wrist_1_rgb", "right_wrist_1_rgb"])
        
        
        # dict_keys(['base_0_rgb', 'base_1_rgb', 'left_wrist_0_rgb', 'left_wrist_1_rgb', 'right_wrist_0_rgb', 'right_wrist_1_rgb'])
        images, img_masks, lang_tokens, lang_masks, state, obs_full = self._preprocess_observation(observation, train=True, return_full_obs=True)
        

        # Normal sampling without fixed seed
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)


        time_expanded = time[:, None, None]
        
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            
            (prefix_output, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],  # * optimizer, gradient.
            )
            return prefix_output, suffix_out


        prefix_output, suffix_out_full = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out_actions = suffix_out_full[:, -self.config.action_horizon :]
        suffix_out_actions = suffix_out_actions.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out_actions)

        # --- Start of Modifications ---

        # Calculate action loss, taking the mean over the action dimension to match JAX implementation
        loss_action = F.mse_loss(u_t, v_t, reduction="none").mean(dim=-1) # Shape: (B, AH)
        loss = loss_action * self.loss_action_weight

        loss_aux_dict = {}
        
        
        if self.with_value_head:
            # Get the state token's final representation
            
            value_loss = torch.zeros(loss.shape[0], 1, device=loss.device)

            deep_rep = suffix_out_full[:, 0, :].to(dtype=torch.float32)
                
            value_pred = self.value_head(deep_rep) # Shape: (B, 1)
            
            if self.TD_learning:
                # * TD learning
                
                # 1. Compute Target Value using Target Network
                with torch.no_grad():
                    # sample_values computes V(next_state)
                    
                    # if use TD_learning, reward specification:
                    # * succesful end (last 10 states): 1
                    # * failure   end (last 10 states): -1
                    # * intermediate: 0
                    cur_frame_index = obs_full.frame_index.float()
                    episode_length = obs_full.episode_length.float()
                    is_failure_data = obs_full.is_failure_data.float()

                    
                    # * within terminal window
                    # * absolute difference between current frame index and episode length is less than or equal to terminal window
                    is_terminal = (cur_frame_index - episode_length).abs() <= self.terminal_window
                    
                    # * is failure data
                    is_failure_data = is_failure_data.float()
                    
                    # * reward
                    # * currently set self.failure_reward to -0.5
                    # * success_reward default 1.0
                    reward = is_terminal * is_failure_data * self.failure_reward + \
                             is_terminal * (1 - is_failure_data) * self.success_reward
                        
                             
                    
                    done = is_terminal.float()

                    # Ensure shapes align
                    if reward.ndim == 1: reward = reward.unsqueeze(1)
                    if done.ndim == 1: done = done.unsqueeze(1)
                    
                    next_value_pred = self.target_model.sample_values(device=actions.device, observation=future_obs)
                    
                    # Bellman Backup: y = r + gamma * (1-d) * Q_targ(s')
                    target_value = reward + self.gamma * (1.0 - done) * next_value_pred

                    if not self.exist_negative_progress:
                        target_value = torch.clamp(target_value, 0.0, self.success_reward)
                    else:
                        target_value = torch.clamp(target_value, -1.0, self.success_reward)
                    
                    
                value_loss += F.mse_loss(value_pred, target_value, reduction="none") * self.loss_value_td_weight
            

            with_progress_loss = random.random() < self.p_with_progress_loss
            if with_progress_loss:
                # * Progress estimate
                
                # Prepare value target (episode progress)
                tgt_frame_index_progress = obs_full.frame_index_progress.float()

                episode_length = obs_full.episode_length.float()
                # Avoid division by zero for episodes of length 0
                episode_length = torch.where(episode_length > 0, episode_length, torch.ones_like(episode_length))
                

                is_failure_data = obs_full.is_failure_data.float()
   
                is_expert_data = 1 - is_failure_data.float()
                if is_expert_data.ndim == 1: is_expert_data = is_expert_data.unsqueeze(1)

                
                if self.exist_negative_progress:
                    progress_tgt = torch.clamp(tgt_frame_index_progress / episode_length, -1.0, 1.0)
                else:
                    progress_tgt = torch.clamp(tgt_frame_index_progress / episode_length, 0., 1.0)  # * tgt_frame_index_progress already processed, might be negative

                progress_tgt = progress_tgt.unsqueeze(1) # Shape: (B, 1)
                
                
                # Calculate value loss
                if self.loss_value_use_bce:
                    value_loss += F.binary_cross_entropy_with_logits(value_pred, progress_tgt, reduction="none") * is_expert_data
                else:
                    value_loss += F.mse_loss(value_pred, progress_tgt, reduction="none") * is_expert_data

            # Weight the value loss
            value_loss = value_loss.to(loss.dtype) * self.loss_value_weight

            # Populate auxiliary dictionary for logging
            loss_aux_dict["loss_action"] = loss_action.detach().mean()
            loss_aux_dict["loss_value"] = value_loss.detach().mean()

            loss = loss + value_loss

        if return_loss_dict:
            return loss, loss_aux_dict

        return loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, 
                       ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""

        # TODO: batch: first half is conditional, second half is unconditional
        bsize = observation.state.shape[0]
        # * for inference, bsize by default is 1.


        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)


        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

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

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt

        
        return x_t

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
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
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
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_values(self, device, observation) -> Tensor:
        """Do a forward pass to compute the value (progress) of the current observation."""
        
        if getattr(observation, 'drop_images', False):
            observation.drop_images(["base_1_rgb", "left_wrist_1_rgb", "right_wrist_1_rgb"])
        
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        bsize = state.shape[0]
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        
        noise_action = self.sample_noise(actions_shape, device)
        time = self.sample_time(bsize, device)
        
        # Embed prefix (images, language) and suffix (state, noisy actions, time)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, noise_action, time)
        
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Perform a single, full forward pass without caching
        (prefix_output, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

       
        deep_rep = suffix_out[:, 0, :].to(dtype=torch.float32)
            

        value_pred = self.value_head(deep_rep)

        # Apply sigmoid if using BCE loss, as the head doesn't have a final activation in that case
        if self.loss_value_use_bce:
            value_pred = torch.sigmoid(value_pred)
            
        return value_pred