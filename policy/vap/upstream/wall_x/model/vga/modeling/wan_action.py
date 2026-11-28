import torch
import torch.nn as nn

from wall_x.model.vga.modeling.wan_dit import (
    CrossAttention,
    SelfAttention,
    precompute_freqs_cis,
)

from wall_x.model.vga.math_utils import (
    _relative2absolute,
    ActionProjector2D,
)


class ActionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        num_views: int = 3,
        view_as_tokens: bool = False,
        state_attn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.state_attn = state_attn
        self.self_attn = SelfAttention(dim, num_heads, eps, enable_sp=False)
        if self.state_attn:
            # Cross-attention to state (proprioception) - action tokens attend to state
            self.cross_attn_state = CrossAttention(
                dim,
                num_heads,
                eps,
                has_image_input=False,
                multi_views=False,
            )
        # Cross-attention to video features (multi-view) — keep name "cross_attn"
        # for backward compatibility with existing checkpoints
        self.cross_attn = CrossAttention(
            dim,
            num_heads,
            eps,
            has_image_input=False,
            multi_views=True,
            view_as_tokens=view_as_tokens,
            num_views=num_views,
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)  # for video cross-attn
        if self.state_attn:
            self.norm4 = nn.LayerNorm(dim, eps=eps)  # for state cross-attn
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = lambda x, g, r: x + g * r

    def forward(
        self,
        a_tokens: torch.Tensor,  # [B, 1+T, dim] - includes state at position 0
        state_tokens,  # [B, 1, dim] or None - state for explicit cross-attention
        video_tokens: torch.Tensor,
        t_mod: torch.Tensor,
        freqs_action,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        # Self-attention (state + actions attend to each other)
        inp = (self.norm1(a_tokens) * (1 + scale_msa)) + shift_msa
        a_tokens = self.gate(a_tokens, gate_msa, self.self_attn(inp, freqs_action))
        if self.state_attn and state_tokens is not None:
            # Cross-attention to state - ensures state info preserved in deeper layers
            a_tokens = a_tokens + self.cross_attn_state(
                self.norm4(a_tokens), state_tokens
            )
        # Cross-attention to video
        a_tokens = a_tokens + self.cross_attn(self.norm3(a_tokens), video_tokens)
        # FFN
        inp = (self.norm2(a_tokens) * (1 + scale_mlp)) + shift_mlp
        a_tokens = self.gate(a_tokens, gate_mlp, self.ffn(inp))
        return a_tokens


class ActionTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        action_dim: int,
        action_embed_dim: int,
        action_mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        video_block_mapping: list = None,
        num_views: int = 3,
        view_as_tokens: bool = False,
        share_state_action_encoder: bool = True,
        local_alignment_pe: bool = False,
        state_attn: bool = False,
        action_per_video_frame: int = 8,  # VAE_temporal_downsample(4) * video_interval
        action_pred_type: str = "v_pred",  # "v_pred"/"x_pred" single head; "v_pred_uni"/"x_pred_uni" dual head
        dit_dim: int = None,  # DiT hidden dim; when different from dim, projection layers are added
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.action_embed_dim = action_embed_dim
        self.local_alignment_pe = local_alignment_pe
        self.state_attn = state_attn
        self.action_per_video_frame = action_per_video_frame
        self.view_as_tokens = view_as_tokens
        # _uni suffix → dual head (primary + second); otherwise single head
        self.dual_head = action_pred_type.endswith("_uni")
        # primary head's pred type: e.g. "x_pred_uni" → "x_pred", "v_pred" → "v_pred"
        self._primary_pred_type = action_pred_type.replace("_uni", "")
        hidden_dim = int(max(action_dim, action_embed_dim) * action_mlp_ratio)
        # Encoder: action -> action_embed
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_embed_dim),
        )

        self.share_state_action_encoder = share_state_action_encoder
        if not self.share_state_action_encoder:
            self.state_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, action_embed_dim),
            )

        # Project embed to transformer dim
        self.proj_in = nn.Linear(action_embed_dim, dim)

        # Projection layers for bridging dit_dim -> action_transformer_dim
        self.dit_dim = dit_dim
        if dit_dim is not None and dit_dim != dim:
            self.video_proj = nn.Linear(dit_dim, dim)
            self.t_mod_proj = nn.Linear(dit_dim, dim)
        else:
            self.video_proj = None
            self.t_mod_proj = None

        if self.local_alignment_pe:
            # Time PE for temporal alignment (action <-> video frame)
            # time_id 0 = obs frame / state, time_id 1+ = subsequent frames
            # max_time_id = max_actions // action_per_video_frame + 1
            self.abs_time_pe = nn.Embedding(64, dim)

        # If no mapping specified, use all video blocks by default (1:1 mapping)
        if video_block_mapping is None:
            self.video_block_mapping = list(range(num_layers))
        else:
            self.video_block_mapping = video_block_mapping

        # Number of action blocks is determined by video_block_mapping length
        actual_num_layers = len(self.video_block_mapping)
        self.blocks = nn.ModuleList(
            [
                ActionBlock(
                    dim,
                    num_heads,
                    ffn_dim,
                    eps,
                    num_views=num_views,
                    view_as_tokens=view_as_tokens,
                    state_attn=self.state_attn,
                )
                for _ in range(actual_num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # Primary head (the only head when dual_head=False; pred type determined by action_pred_type)
        self.head = nn.Linear(dim, action_embed_dim)  # 1536, 768
        self.action_decoder = nn.Sequential(
            nn.Linear(action_embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Secondary head (only created in _uni mode for the other pred type)
        if self.dual_head:
            self.head_second = nn.Linear(dim, action_embed_dim)
            self.action_decoder_second = nn.Sequential(
                nn.Linear(action_embed_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, action_dim),
            )

    def _get_action_time_ids(self, T, device, n_state=1):
        """Get time_ids for action sequence [state_0, ..., state_{N-1}, action_0, ..., action_{T-1}]
        state_0 (oldest) -> 0, state_1 -> 1, ..., state_{N-1} (newest) -> N-1,
        action[0:8] -> N, action[8:16] -> N+1, ...
        """
        # State tokens: oldest=0, newest=n_state-1
        state_ids = torch.arange(n_state, dtype=torch.long, device=device)
        # Action tokens: continue from n_state
        action_ids = (
            torch.arange(T, device=device) // self.action_per_video_frame + n_state
        )
        return torch.cat([state_ids, action_ids])  # [n_state+T]

    def _get_video_time_ids(self, f, hw, device):
        """Get time_ids for video sequence [frame_0, frame_1, ..., frame_{f-1}]
        Each frame has hw spatial tokens, all sharing the same time_id.
        frame_0 (obs) -> 0, frame_1 -> 1, frame_2 -> 2, ...
        """
        # [f] -> repeat each for hw tokens -> [f*hw]
        frame_ids = torch.arange(f, device=device)
        return frame_ids.repeat_interleave(hw)  # [f*hw]

    def forward(
        self,
        noisy_actions: torch.Tensor,
        agent_pos: torch.Tensor,
        video_features_per_block: list[torch.Tensor],
        t_mod: torch.Tensor,
        action_history: torch.Tensor,
        active_head: str = "both",
    ):
        """
        Args:
            noisy_actions: [B_action, T, action_dim]
            agent_pos: [B_action, N_state, action_dim]  (N_state >= 1)
            action_history_expanded: [B_action, T_history, action_dim]
            video_features_per_block: list of [B_video, f, h, w, dim]
            t_mod: [B_action, 6, dim]
            active_head: Which head(s) to compute. Only relevant when dual_head=True.
                - "both": compute both v_pred and x_pred heads (for training)
                - "v_pred": compute only the primary (v_pred) head
                - "x_pred": compute only the secondary (x_pred) head
        """
        B_action, T, _ = noisy_actions.shape
        N_state = agent_pos.shape[1]
        device = noisy_actions.device

        # Encode actions and (optionally) states
        action_embed = self.action_encoder(
            noisy_actions
        )  # [B_action, T, action_embed_dim]
        if self.share_state_action_encoder:
            state_embed = self.action_encoder(
                agent_pos
            )  # [B_action, N_state, action_embed_dim]
        else:
            state_embed = self.state_encoder(
                agent_pos
            )  # [B_action, N_state, action_embed_dim]
        if action_history is not None:
            action_history_embed = self.action_encoder(action_history)

        if action_history is not None:
            history_future_action_embed = torch.cat(
                [action_history_embed, action_embed], dim=1
            )  # [B_action, N_state+T, action_embed_dim]
            a = self.proj_in(history_future_action_embed)  # [B_action, N_state+T, dim]
        else:
            state_action_embed = torch.cat(
                [state_embed, action_embed], dim=1
            )  # [B_action, N_state+T, action_embed_dim]
            a = self.proj_in(state_action_embed)  # [B_action, N_state+T, dim]

        if self.local_alignment_pe:
            # Add time PE to action tokens (for cross-attn alignment)
            action_time_ids = self._get_action_time_ids(T, device, n_state=N_state)
            a = a + self.abs_time_pe(
                action_time_ids
            )  # broadcast: [B, N_state+T, dim] + [N_state+T, dim]

        # Extract state tokens for explicit cross-attention
        state_tokens = None
        if self.state_attn:
            if action_history is not None:
                state_tokens = self.proj_in(state_embed)
            else:
                state_tokens = a[:, :N_state, :].clone()  # [B_action, N_state, dim]

        # RoPE for action tokens (for self-attn position)
        head_dim = self.dim // self.blocks[0].self_attn.num_heads
        # Build cos/sin freqs for xcompute rope kernel directly on action device
        cos_action, sin_action = precompute_freqs_cis(
            head_dim, end=a.shape[1], device=device
        )
        cos_action = cos_action.view(1, a.shape[1], -1)
        sin_action = sin_action.view(1, a.shape[1], -1)
        freqs_action = (cos_action, sin_action)

        # Project timestep modulation from dit_dim to action_transformer_dim
        if self.t_mod_proj is not None:
            t_mod = self.t_mod_proj(t_mod)

        # Iterate blocks
        for blk_idx, blk in enumerate(self.blocks):
            video_blk_idx = self.video_block_mapping[blk_idx]
            vt = video_features_per_block[
                min(video_blk_idx, len(video_features_per_block) - 1)
            ]
            # Project video features from dit_dim to action_transformer_dim
            if self.video_proj is not None:
                vt = self.video_proj(vt)
            B_video, f, h, w, dim_feat = vt.shape
            hw = h * w
            num_views = self.blocks[0].cross_attn.num_views  # fixed from model config
            B_data = B_video // num_views
            view_as_tokens = getattr(self.blocks[0].cross_attn, "view_as_tokens", False)

            if self.local_alignment_pe:
                # Add time PE to video tokens before flattening
                video_time_ids = self._get_video_time_ids(f, hw, device)  # [f*hw]
                video_time_pe = self.abs_time_pe(video_time_ids)  # [f*hw, dim]

                # Flatten: [B_video, f, h, w, dim] -> [B_video, f*hw, dim]
                vt_flat = vt.reshape(B_video, f * hw, dim_feat)
                vt_flat = vt_flat + video_time_pe  # add time PE

                # Reshape to [B_data, num_views, f*hw, dim]
                vt_flat = vt_flat.reshape(B_data, num_views, f * hw, dim_feat)
                if view_as_tokens:
                    vt_flat = vt_flat.reshape(B_data, num_views * f * hw, dim_feat)
                else:
                    # Concat views in channel dim: [B_data, f*hw, dim*num_views]
                    vt_flat = vt_flat.permute(0, 2, 1, 3).reshape(
                        B_data, f * hw, dim_feat * num_views
                    )
            else:
                # Flatten spatial-temporal features: [B*num_views, f, h, w, dim] -> [B, f*h*w, dim*num_views]
                vt_flat = vt.reshape(B_data, num_views, f * h * w, dim_feat)
                if view_as_tokens:
                    vt_flat = vt_flat.reshape(B_data, num_views * f * h * w, dim_feat)
                else:
                    vt_flat = vt_flat.permute(0, 2, 1, 3).reshape(
                        B_data, f * h * w, dim_feat * num_views
                    )

            if B_action > B_data:
                vt_flat = vt_flat.repeat(
                    B_action // B_data, 1, 1
                )  # expand for action_batch_size

            a = blk(a, state_tokens, vt_flat, t_mod, freqs_action)

        a_normed = self.norm_out(a)

        if self.dual_head:
            # Dual-head mode: compute only the requested head(s) to save compute
            if active_head != "both":
                # Inference: run only the requested head
                # Determine which physical head matches the requested pred type
                # primary head = action_pred_type without _uni suffix,
                # second head = the other pred type
                _primary_type = self._primary_pred_type
                if active_head == _primary_type:
                    out = self.action_decoder(self.head(a_normed))[:, N_state:, :]
                else:
                    out = self.action_decoder_second(self.head_second(a_normed))[
                        :, N_state:, :
                    ]
                return out
            else:
                # Training: compute both heads, return (primary_out, second_out)
                out_primary = self.action_decoder(self.head(a_normed))[:, N_state:, :]
                out_second = self.action_decoder_second(self.head_second(a_normed))[
                    :, N_state:, :
                ]
                return out_primary, out_second

        # Single-head mode (non-_uni): always use primary head
        a_embed_out = self.head(a_normed)
        return self.action_decoder(a_embed_out)[:, N_state:, :]


class ActionConditioner(nn.Module):
    """Encode ground-truth actions into context tokens for DiT cross-attention."""

    def __init__(
        self,
        dim=1536,
        checkpoint_path=None,
        device=None,
        calib_json_path=None,
        target_height=480,
        target_width=640,
        gripper_in_eef=(0.162, 0.0, 0.0),
    ):
        super().__init__()

        from x2robot_tokenizer.x2robot_tokenizer_v3_1_delta.api import (
            X2RobotTokenizerV3_1,
        )

        self.action_tokenizer = X2RobotTokenizerV3_1(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        ACTION_TOKEN_DIM = 1024  # should match tokenizer's output dimension
        INFO2D_DIM = 8  # should match projector's output dimension
        self.proj1 = nn.Linear(ACTION_TOKEN_DIM, dim)
        self.proj2 = nn.Linear(INFO2D_DIM, dim)

        self.projector_2d = None
        if calib_json_path is not None:
            self.projector_2d = ActionProjector2D(
                calib_json_path,
                target_height=target_height,
                target_width=target_width,
                gripper_in_eef=gripper_in_eef,
            )

    def _apply(self, fn, recurse=True):
        """
        Override `_apply` to also migrate `action_tokenizer`, which is not an nn.Module.

        PyTorch's recursive device migration (triggered by .to()/.cuda()/accelerator.prepare())
        calls `_apply(fn)` on each registered nn.Module — but skips plain Python objects.
        X2RobotTokenizerV3_1 is a plain class holding internal torch modules, so we must
        manually forward the _apply call to its internals.
        """
        ret = super()._apply(fn, recurse=recurse)

        tok = getattr(self, "action_tokenizer", None)
        if tok is not None:
            if hasattr(tok, "model") and tok.model is not None:
                tok.model._apply(fn)
            if hasattr(tok, "vision_projector") and tok.vision_projector is not None:
                tok.vision_projector._apply(fn)
            tok.device = self.proj1.weight.device

        projector_2d = getattr(self, "projector_2d", None)
        if projector_2d is not None and hasattr(projector_2d, "_apply"):
            projector_2d._apply(fn)

        return ret

    def forward(self, actions, agent_pos=None, dof_mask=None):
        """
        Args:
            actions: [B, T, action_dim]
            agent_pos: [B, 1, action_dim] or None
            dof_mask: [B, T, action_dim] or None
        Returns:
            [B, token_num, dim] - action context tokens for DiT cross-attention
        """
        # test
        # self.test_reconstruct(actions, agent_pos, dof_mask)

        result = self.action_tokenizer.encode(
            actions=actions,
            obs_state=agent_pos,
            dof_mask=dof_mask,
        )
        tokens = result["quantized_features"]
        action_context = self.proj1(tokens)
        context2d = self.project2d(actions, agent_pos)
        return torch.cat([action_context, context2d], dim=1)

    def test_reconstruct(self, actions, agent_pos, dof_mask):
        results = self.action_tokenizer.reconstruct(
            actions=actions,
            obs_state=agent_pos,
            dof_mask=dof_mask,
        )
        reconstructed_actions = results["reconstructed"]
        L1_loss = torch.abs(reconstructed_actions - actions).mean()
        THRESHOLD = 5e-2
        if L1_loss > THRESHOLD:
            print(
                f"Reconstruction Test Failed: {L1_loss.item():.4f} exceeds threshold {THRESHOLD}"
            )
        else:
            print("Reconstruction Test Passed")

    def project2d(self, actions: torch.Tensor, agent_pos: torch.Tensor) -> dict:
        """Project absolute 14-dim actions to per-camera 2-D pixel coordinates.

        Args:
            abs_actions: [B, T, action_dim], delta action
            agent_pos: [B, 1, action_dim], agent position
            cameras: currently face-only projection

        Returns:
            dict:
                "face_view":        [B, T, 4]  (left_u, left_v, right_u, right_v)
                "left_wrist_view":  [B, T, 2]  (left_u, left_v)
                "right_wrist_view": [B, T, 2]  (right_u, right_v)
        """
        if self.projector_2d is None:
            raise RuntimeError(
                "ActionProjector2D not initialised. "
                "Call init_projector(calib_json_path) first."
            )
        abs_actions = _relative2absolute(actions, agent_pos)
        result = self.projector_2d.project(abs_actions, return_eef=2)
        gripper = result["gripper"]["face_view"]
        eef = result["eef"]["face_view"]
        info2d = torch.cat([gripper, eef], dim=-1)
        context2d = self.proj2(info2d)
        return context2d
