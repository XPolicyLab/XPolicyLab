import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from skyrobot.models.attention import WanLayerNorm
from skyrobot.models.attention_block import WanAttentionBlock
from skyrobot.models.attn_op import FlexAttnFunc
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from skyrobot.models.rotary import sinusoidal_embedding_1d, WanRotaryPosEmbed
from skyrobot.data.data_config import MAX_ACTION_DIM, MAX_STATE_DIM
from skyrobot.data.data_config import EmbodimentId


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        dt = x.dtype
        e = (self.modulation.float().unsqueeze(0) + e.float().unsqueeze(2)).chunk(2, dim=2)
        x = self.norm(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        x = self.head(x.to(dt))
        return x


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 注册 empty 参数占位。**真正的 init 在 reset_parameters 里完成**,
        # 这样 torchtitan 的 meta-init → to_empty → _init_all_weights 流程能正确
        # 调用到我们这里的 init(通过 hasattr(module, 'reset_parameters'))。
        # 如果只在 __init__ 里随机,meta 上下文里 torch.randn 会返回 meta 占位,
        # to_empty 之后就是未初始化的垃圾内存(CUDA 上通常是 0),前向死掉,loss 不降。
        self.W = nn.Parameter(torch.empty(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.empty(num_categories, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """DreamZero 的 stddev=0.02 初始化。"""
        with torch.no_grad():
            self.W.normal_(0.0, 0.02)
            self.b.zero_()

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class Action_Head(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim, eps=1e-6):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, output_dim)
        #self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)
        # modulation
        self.norm = WanLayerNorm(input_dim, eps)
        self.modulation = nn.Parameter(torch.randn(1, 2, input_dim) / input_dim**0.5)

    def forward(self, x, cat_ids, e):
        #hidden = F.relu(self.layer1(x, cat_ids))
        #return self.layer2(hidden, cat_ids)
        dt = x.dtype
        e = (self.modulation.float().unsqueeze(0) + e.float().unsqueeze(2)).chunk(2, dim=2)
        x = self.norm(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        return self.layer1(x.to(dt), cat_ids)



class SkyrobotWAM(ModelMixin, ConfigMixin):

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=48,
        dim=3072,
        ffn_dim=14336,
        freq_dim=256,
        text_dim=4096,
        action_dim=MAX_ACTION_DIM,
        action_per_latent=16,
        chunk_size=1,
        state_dim=MAX_STATE_DIM,
        out_dim=48,
        num_heads=16,
        num_layers=30,
        rope_max_seq_len=1024,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        num_embodiments=1,
        supported_embodiments=None,   # list[EmbodimentId];None 时默认 [EmbodimentId(0)]
    ):
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.action_per_latent = action_per_latent
        self.state_dim = state_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_layers = num_layers
        self.rope_max_seq_len = rope_max_seq_len
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_embodiments = num_embodiments
        if supported_embodiments is None:
            supported_embodiments = [EmbodimentId(0)]
        # 存 list[int],避免 enum 序列化/迁移问题,与 batch 里的 int64 tensor 类型对齐
        self.supported_embodiments = [int(e) for e in supported_embodiments]
        assert len(self.supported_embodiments) == num_embodiments, (
            f"supported_embodiments has {len(self.supported_embodiments)} items "
            f"but num_embodiments={num_embodiments}"
        )
        self.embodiments_table = self.embodiment_id_to_slot()

        # embeddings
        # text embeddings
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        # latent embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # action embeddings
        self.action_embedding = CategorySpecificLinear(self.num_embodiments, action_dim, dim)
        self.action_time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.action_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # state embeddings
        self.state_embedding = CategorySpecificLinear(self.num_embodiments, state_dim, dim)
        self.state_time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.state_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # positional embeddings
        self.rope = WanRotaryPosEmbed(self.head_dim, patch_size, rope_max_seq_len)

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, qk_norm,
                cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        self.action_head = Action_Head(self.num_embodiments, dim, dim*2, action_dim)
        self._gradient_checkpointing = False

    def embodiment_id_to_slot(self) -> dict:
        return {int(e): i for i, e in enumerate(self.supported_embodiments)}

    def embodiment_id_to_category_id(self, embodiment_id: torch.Tensor) -> torch.Tensor:
        """把 batch 里 raw EmbodimentId 值 → CategorySpecificLinear 的 slot 下标。

        Args:
            embodiment_id: [B] long
        Returns:
            cat_ids: [B] long,∈ [0, num_embodiments)
        Raises:
            AssertionError:batch 里出现未在 supported_embodiments 声明的 id
        """
        mapping = self.embodiments_table
        # 用 -1 做哨兵,循环结束后检查是否还有 -1(= 未识别的 raw id)
        cat_ids = torch.full_like(embodiment_id, -1)
        for raw_id, slot_id in mapping.items():
            cat_ids[embodiment_id == raw_id] = slot_id
        assert (cat_ids >= 0).all(), (
            f"batch 里有未在 supported_embodiments={self.supported_embodiments} "
            f"里的 embodiment_id: "
            f"{embodiment_id[cat_ids < 0].unique().tolist()}"
        )
        return cat_ids

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False

    def _text_embed(self, text):
        return self.text_embedding(text)
    def _latent_embed(self, latents):
        latents = self.patch_embedding(latents)
        latents = latents.flatten(2).transpose(1, 2)
        return latents
    def _action_embed(self, actions, cat_ids):
        actions = rearrange(actions, 'b c f h w -> b (f h w) c')
        return self.action_embedding(actions, cat_ids)
    def _state_embed(self, states, cat_ids):
        states = rearrange(states, 'b c f h w -> b (f h w) c')
        return self.state_embedding(states, cat_ids)

    def _time_latent_embed(self, timesteps, H, W, batch_size):
        pach_scale_h, pach_scale_w = self.patch_size[1], self.patch_size[2]
        time_steps = torch.repeat_interleave(
            timesteps, (H // pach_scale_h) * (W // pach_scale_w), dim=1
        )
        # print('Latent Time Steps Shape:', time_steps.shape)
        time_emb = sinusoidal_embedding_1d(
            self.freq_dim, time_steps
        ).type(self.time_embedding[0].weight.dtype)
        # print(time_emb.shape)
        time_emb = self.time_embedding(time_emb)
        timestep_proj = self.time_projection(time_emb)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        return time_emb, timestep_proj

    def _time_action_embed(self, timesteps, H, W, batch_size):
        pach_scale_h, pach_scale_w = (1, 1)
        time_steps = torch.repeat_interleave(
            timesteps, (H // pach_scale_h) * (W // pach_scale_w), dim=1
        )
        # print('Action Time Steps Shape:', time_steps.shape)
        time_emb = sinusoidal_embedding_1d(
            self.freq_dim, time_steps
        ).type(self.action_time_embedding[0].weight.dtype)
        # print(time_emb.shape)
        time_emb = self.action_time_embedding(time_emb)
        timestep_proj = self.action_time_projection(time_emb)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        return time_emb, timestep_proj

    def _time_state_embed(self, timesteps, H, W, batch_size):
        pach_scale_h, pach_scale_w = (1, 1)
        time_steps = torch.repeat_interleave(
            timesteps, (H // pach_scale_h) * (W // pach_scale_w), dim=1
        )
        # print('State Time Steps Shape:', time_steps.shape)
        time_emb = sinusoidal_embedding_1d(
            self.freq_dim, time_steps
        ).type(self.state_time_embedding[0].weight.dtype)
        # print(time_emb.shape)
        time_emb = self.state_time_embedding(time_emb)
        timestep_proj = self.state_time_projection(time_emb)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        return time_emb, timestep_proj


    def _reconstruct_latent(self, latent_seq, input_dict):
        """Reshape latent sequence [B, L, C*prod(patch)] back to [B, C, F, H, W]."""
        C, _, H, W = input_dict['latent_dict']['latent_shapes'][0]
        pf, ph, pw = self.patch_size
        h, w = H // ph, W // pw
        f = latent_seq.shape[1] // (h * w)  # derive F from tensor, not per-sample shape
        return rearrange(
            latent_seq,
            'b (f h w) (pf ph pw c) -> b c (f pf) (h ph) (w pw)',
            f=f, h=h, w=w,
            pf=pf, ph=ph, pw=pw,
        )

    def _reconstruct_action(self, action_seq, input_dict):
        """Reshape action sequence back to [B, C_action, F, action_per_latent, 1]."""
        action_shapes = input_dict['action_dict']['action_shapes']
        h, w = action_shapes[0][2], action_shapes[0][3]  # apf and 1, same for all samples
        f = action_seq.shape[1] // (h * w)  # derive F from tensor
        return rearrange(
            action_seq,
            'b (f h w) c -> b c f h w',
            f=f, h=h, w=w,
        )

    def forward(self, input_dict):

        batch_size = input_dict['latent_dict']['condation_latents'].shape[0]

        condation_latents = input_dict['latent_dict']['condation_latents']
        noisy_latents = input_dict['latent_dict']['noisy_latents']
        clean_latents = input_dict['latent_dict']['latent']
        noisy_actions = input_dict['action_dict']['noisy_latents']
        clean_actions = input_dict['action_dict']['latent']
        states = input_dict['state_dict']['latent']
        text_embed = input_dict['text_dict']['text']

        embodiment_ids = input_dict['embodiment_ids']  # Assuming embodiment_ids is [B]
        cat_ids = self.embodiment_id_to_category_id(embodiment_ids)  # Use embodiment_ids directly as category IDs for embeddings

        condation_latents_embed = self._latent_embed(condation_latents)
        clean_latents_embed = self._latent_embed(clean_latents)
        noisy_latents_embed = self._latent_embed(noisy_latents)
        clean_actions_embed = self._action_embed(clean_actions, cat_ids)
        noisy_actions_embed = self._action_embed(noisy_actions, cat_ids)
        states_embed = self._state_embed(states, cat_ids)
        text_embed = self._text_embed(text_embed)

        # print("Condation Latents Shape:", condation_latents_embed.shape)
        # print("Clean Latents Shape:", clean_latents_embed.shape)
        # print("Noisy Latents Shape:", noisy_latents_embed.shape)
        # print("Clean Actions Shape:", clean_actions_embed.shape)
        # print("Noisy Actions Shape:", noisy_actions_embed.shape)
        # print("States Embed Shape:", states_embed.shape)
        # print("Text Embed Shape:", text_embed.shape)

        latent_time_steps = torch.cat([
            input_dict['latent_dict']['condation_latent_timesteps'],
            input_dict['latent_dict']['clean_latent_timesteps'],
            input_dict['latent_dict']['noisy_latent_timesteps']
        ],dim=1)
        action_time_steps = torch.cat([
            input_dict['action_dict']['clean_action_timesteps'],
            input_dict['action_dict']['noisy_action_timesteps']
        ],dim=1)
        state_time_steps = input_dict['state_dict']['latent_timesteps']
        # print('latent_time_steps shape', latent_time_steps.shape)
        # print('action_time_steps shape', action_time_steps.shape)
        # print('state_time_steps shape', state_time_steps.shape)
        latent_temb, latent_timestep_proj =self._time_latent_embed(
            latent_time_steps,
            input_dict['latent_dict']['condation_latents'].shape[-2],
            input_dict['latent_dict']['condation_latents'].shape[-1],
            batch_size
        )
        action_temb, action_timestep_proj =self._time_action_embed(
            action_time_steps,
            input_dict['action_dict']['noisy_latents'].shape[-2],
            input_dict['action_dict']['noisy_latents'].shape[-1],
            batch_size
        )
        _, state_time_proj = self._time_state_embed(
            state_time_steps,
            input_dict['state_dict']['latent'].shape[-2],
            input_dict['state_dict']['latent'].shape[-1],
            batch_size
        )
        # print('Latent Temb Shape:', latent_temb.shape)
        # print('Latent Timestep Proj Shape:', latent_timestep_proj.shape)
        # print('Action Timestep Proj Shape:', action_timestep_proj.shape)
        # print('State Timestep Proj Shape:', state_time_proj.shape)


        hidden_states = torch.cat([
            condation_latents_embed,
            clean_latents_embed,
            noisy_latents_embed,
            states_embed,
            clean_actions_embed,
            noisy_actions_embed
        ], dim=1)
        # print('hidden_states shape', hidden_states.shape)

        timestep_proj = torch.cat([
            latent_timestep_proj,
            state_time_proj,
            action_timestep_proj],
        dim=1)
        # print('Timestep Proj Shape:', timestep_proj.shape)


        clean_latent_grid_id = input_dict['latent_dict']['clean_latent_grid_id']
        noisy_latent_grid_id = input_dict['latent_dict']['noisy_latent_grid_id']
        action_grid_id = input_dict['action_dict']['action_grid_id']
        state_grid_id = input_dict['state_dict']['state_grid_id']

        # print(clean_latent_grid_id.shape)
        # print(noisy_latent_grid_id.shape)
        # print(state_grid_id.shape)
        # print(action_grid_id.shape)
        # print(action_grid_id.shape)

        full_grid_id = torch.cat([clean_latent_grid_id] + [noisy_latent_grid_id] + [state_grid_id] + [action_grid_id] * 2, dim=2)
        # print(full_grid_id.shape)
        rotary_emb = self.rope(full_grid_id)[:, :, None]
        # print('rotary_emb shape', rotary_emb.shape)

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128

        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [
            condation_latents_embed.shape[1],
            clean_latents_embed.shape[1],
            noisy_latents_embed.shape[1],
            states_embed.shape[1],
            clean_actions_embed.shape[1],
            noisy_actions_embed.shape[1],
            padded_length
        ]

        #if FlexAttnFunc.attention_mask is None or FlexAttnFunc.attention_mask.shape[0] != batch_size:
        FlexAttnFunc.init_mask(
            context_shapes=input_dict['latent_dict']['context_shapes'],
            latent_shapes=input_dict['latent_dict']['latent_shapes'],
            state_shapes=input_dict['state_dict']['state_shapes'],
            action_shapes=input_dict['action_dict']['action_shapes'],
            chunk_size=self.chunk_size,
            patch_size=self.patch_size,
            device=hidden_states.device,
            text_len=self.text_len,
            text_mask=input_dict['text_dict'].get('text_mask', None),
        )

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, text_embed, timestep_proj, rotary_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    text_embed,
                    timestep_proj,
                    rotary_emb,
                )
        _, _, latent_hidden_states, _, _, action_hidden_states, _ = torch.split(hidden_states, split_list, dim=1)
        _, _, latent_temb = torch.split(latent_temb, split_list[:3], dim=1)
        _, action_temb = torch.split(action_temb, split_list[4:6], dim=1)
        latent_hidden_states = self.head(latent_hidden_states, latent_temb)
        action_hidden_states = self.action_head(action_hidden_states, cat_ids, action_temb)

        latent_hidden_states = self._reconstruct_latent(latent_hidden_states, input_dict)
        action_hidden_states = self._reconstruct_action(action_hidden_states, input_dict)

        return latent_hidden_states, action_hidden_states


    @classmethod
    def from_wan_pretrained(cls, ckpt_dir, model_config=None, dtype=torch.bfloat16, device='cpu', **kwargs):
        """Create a SkyrobotWAM model and load matching weights from a Wan2.2 checkpoint.

        Args:
            ckpt_dir: Path to directory containing *.safetensors files.
            model_config: SkyrobotWAMConfig instance. If provided, its fields are used as model kwargs.
            dtype: Model dtype (default: bfloat16).
            device: Device to place the model on after loading.
            **kwargs: Extra kwargs that override model_config fields.

        Returns:
            model: SkyrobotWAM with pretrained weights loaded for all matching keys.
        """
        import os
        import glob
        from dataclasses import asdict
        from safetensors.torch import load_file
        from skyrobot.utils.logging import rank0_info, rank0_warning

        # Merge model_config and kwargs; kwargs take precedence
        if model_config is not None:
            config_kwargs = asdict(model_config)
            config_kwargs.update(kwargs)
        else:
            config_kwargs = kwargs

        rank0_info(f"Loading Wan pretrained weights from {ckpt_dir}")
        model = cls(**config_kwargs).to(dtype)

        # Load all safetensor shards, cast to target dtype to avoid double memory
        pretrained_state = {}
        for f in sorted(glob.glob(os.path.join(ckpt_dir, '*.safetensors'))):
            for k, v in load_file(f, device='cpu').items():
                pretrained_state[k] = v.to(dtype)

        # Classify keys
        model_state = model.state_dict()
        matched, mismatched, missing, extra = [], [], [], []

        for key in model_state:
            if key in pretrained_state:
                if model_state[key].shape == pretrained_state[key].shape:
                    matched.append(key)
                else:
                    mismatched.append((key, model_state[key].shape, pretrained_state[key].shape))
            else:
                missing.append(key)

        for key in pretrained_state:
            if key not in model_state:
                extra.append(key)

        # Load matched keys and free pretrained_state
        model.load_state_dict({k: pretrained_state[k] for k in matched}, strict=False)
        del pretrained_state

        rank0_info("=== Wan Pretrained Weight Loading Summary ===")
        rank0_info(f"  Matched & loaded:        {len(matched)}")
        rank0_info(f"  Shape mismatch:          {len(mismatched)}")
        rank0_info(f"  In model but not ckpt:   {len(missing)}")
        rank0_info(f"  In ckpt but not model:   {len(extra)}")
        if mismatched:
            for k, ms, ps in mismatched:
                rank0_warning(f"    Shape mismatch: {k}: model={ms}, ckpt={ps}")
        if missing:
            for k in missing:
                rank0_info(f"    New layer (random init): {k}")
        if extra:
            for k in extra:
                rank0_info(f"    Unused ckpt key: {k}")

        model = model.to(device)
        return model
