# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from .architecture_contract import (
    ALLOWED_ARCHITECTURE_VERSIONS,
    LEGACY_ARCHITECTURE_VERSION,
    canonicalize_action_context_mode,
    canonicalize_architecture_version,
    canonicalize_fast_visibility_policy,
    canonicalize_prefix_layout_version,
    canonicalize_semantic_anchor_policy,
    is_prefix_kv_action_context_mode,
    is_vlm_fast_action_bridge_version,
    normalize_condition_source,
    reject_removed_suffix_adapter,
    resolve_architecture_path,
)
from .configuration_florence2 import Florence2Config
from transformers.configuration_utils import PretrainedConfig


class XVLAConfig(PretrainedConfig):
    """
    Configuration class for the **XVLA (Extended Vision-Language-Action)** model.

    This configuration defines all submodules of XVLA in a single place:
      - The visual-language backbone (Florence2)
      - The temporal/action transformer
      - The action/proprio setup
    """

    model_type = "xvla"

    def __init__(
        # === Florence backbone ===
        self,
        florence_config: dict | None = None,
        # Semantic VLM plugin. ``florence_config`` remains serialized even for
        # non-Florence backends because the independent UAM Dorsal encoder is
        # intentionally a DaViT expert, not a copy of the semantic VLM.
        vlm_backend: str = "florence2",
        vlm_config: dict | None = None,
        vlm_model_name_or_path: str = "",
        vlm_projection_dim: int = 0,
        vlm_prompt_config: dict | None = None,
        vlm_dtype: str = "auto",
        vlm_lora: dict | None = None,
        vlm_max_view_embeddings: int = 8,
        # Qwen-only visual history encoding. ``off`` preserves the legacy
        # masked frame average; fixed_sinusoidal_v1 adds deterministic frame
        # positions before history reduction. ``spacetime_mem_v1`` opts into
        # the MEM-style causal cross-frame attention inside the Qwen ViT
        # (see ``models/qwen_spacetime.py``); it is a strict opt-in and
        # requires ``dataset.img_history_size_per_view`` with at least one
        # K>1 view.
        qwen_temporal_embedding: str = "off",
        # Scheme A only: opt into bidirectional attention inside each image
        # block. The *_current_only mode additionally applies proposal (b)
        # and restricts G0.5 visual prefix K/V to current images, like 2B.
        qwen_image_attention_mode: str = "causal",
        # Training-only frozen-teacher residual vision objective; None is off.
        qwen_far_aux: dict | None = None,
        qwen_history_time_rope: dict | None = None,
        # ``spacetime_mem_v1`` stride: wrap every ``qwen_spacetime_stride``-th
        # vision block with the MEM patch. Default 4 matches the Hy-VLA
        # original and the Florence DaViT port; ignored under ``off`` /
        # ``fixed_sinusoidal_v1``.
        qwen_spacetime_stride: int = 4,
        # ``vlm_history_value_v1`` data contract: dataset steps between head
        # history slots. The temporal frame code itself is indexed in slot
        # units, so this value does not scale it; it exists so training can
        # assert the model and ``dataset.img_history_interval`` agree, and so
        # a checkpoint records the spacing its history was trained on.
        qwen_history_value_interval: int = 40,
        # Rotary base of the ``vlm_history_value_v1`` temporal branch. 10000 is
        # the measured sweet spot: it raises injected temporal-order
        # information without reducing content sensitivity or the oldest
        # frame's weight. 100 injects more order information but halves the
        # oldest-frame weight. See
        # docs/history_probe_separability_2026-10-20.md section 10.
        qwen_history_value_rope_theta: float = 10000.0,
        # DM0.5-inspired explicit history bank. The existing temporal encoder
        # still produces the current-view features; this optional branch keeps
        # prior frames as independent pooled spatial tokens for the action
        # expert's separate read-only memory attention.
        qwen_history_memory_enabled: bool = False,
        qwen_history_memory_pool_size: int = 4,
        qwen_history_memory_gate_init: float = 0.0,
        # P3-style history through Qwen multimodal packing. Past frames are
        # independently pooled to 16 tokens and emitted as ordered T=1 image
        # blocks; the current frame keeps the 49-token contract. Fixed
        # sinusoidal frame positions and text timestamps are retained, while
        # ViT-level spacetime_mem_v1 remains incompatible with this path.
        #
        # ``qwen_native_video_history_views`` names *raw camera* view indices,
        # i.e. the pre-``_reorder_visual_features`` order the dataset emits
        # (head=0, left_wrist=1, right_wrist=2 for Hy-VLA). The reorder helper
        # rotates the first *valid* view into slot 0 as a packing
        # normalization, not a semantic identity, so it must not be used to
        # interpret this option. When a named view is masked out for a sample
        # no view expands for that sample and the single-frame contract
        # applies. Empty tuple leaves the K=1 path unchanged.
        #
        # Each expanded view is capped at max(grid)//spatial_merge_size = 7
        # frames; see QWEN_POLICY_MAX_NATIVE_VIDEO_FRAMES for why.
        qwen_native_video_history_enabled: bool = False,
        qwen_native_video_history_views: tuple[int, ...] | list[int] | None = None,
        # Zero-initialized, scale-only AdaLayerNorm for Qwen action tokens.
        # Disabled by default for checkpoint parity.
        qwen_action_adalayernorm_enabled: bool = False,
        dm05_action_config: dict | None = None,
        # Number of native DM05 state/action channels serialized in the prompt.
        # The public Hy batch remains padded to 32 channels.
        dm05_state_dim: int = 20,

        # === Transformer head ===
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 30,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        # Flow-matching timestep embedding conditioning. The defaults reproduce
        # the original X-VLA parameterization bit-exactly, which is badly
        # conditioned for t in [0.001, 1] (see models/transformer.py's
        # ``timestep_embedding`` docstring). Set max_period=10000 with
        # input_scale=1000 for the standard, well-conditioned variant.
        time_embedding_max_period: int = 100,
        time_embedding_input_scale: float = 1.0,
        # Flow-matching timestep sampling distribution. The default reproduces
        # the original X-VLA / Hy-VLA / pi0.5 recipe (shifted Beta(1.5, 1.0)
        # over [0.001, 1.000]). ``logit_normal`` swaps in the Logit-Normal
        # schedule proposed by FACT (arXiv 2608.01402); the location m biases
        # sampling toward low t (precision regime) while s controls the mass
        # remaining at high t (needed for the first Euler step at inference).
        # See docs/flow_time_distribution_ln_probe_2026-09-11.md for the
        # motivation and the FACT paper's blind spot near t~1.
        flow_time_distribution: str = "beta",
        # Shape parameters for the shifted Beta schedule.  The historical
        # values are kept as defaults so old checkpoints remain bit-exact.
        beta_a: float = 1.5,
        beta_b: float = 1.0,
        logit_normal_m: float = 1.5,
        logit_normal_s: float = 1.0,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        soft_prompt_length: int = 32,
        # ``auto`` selects the semantic backend's compatible policy:
        # Florence keeps the historical joint sequence, while Qwen exposes a
        # read-only multimodal context consumed by action-query cross-attention.
        action_context_mode: str = "auto",
        action_context_cross_attn_every_n_layers: int = 4,
        action_context_cross_attn_gate_init: float = 0.1,
        action_context_trained: bool = False,
        # --- vlm_fast_action_bridge (G0.5 prefix-KV) contract ---------------
        # Every field below defaults to the historical legacy value, so a
        # checkpoint that omits all of them resolves to the unchanged fused
        # Florence/Qwen MHA path. They are only read when a checkpoint
        # explicitly declares architecture_version=vlm_fast_action_bridge.
        architecture_version: str = LEGACY_ARCHITECTURE_VERSION,
        architecture_path: str | None = None,
        cache_topology: str | None = None,
        condition_state_version: str = "none",
        prefix_layout_version: str = "legacy_v5",
        semantic_anchor_policy: str = "quarter_uniform",
        fast_visibility_policy: str = "legacy_semantic_capture_v1",
        matched_dual_stream_fast_ce_version: str = "none",
        matched_dual_stream_route: list[int] | tuple[int, ...] | None = None,
        # GQA geometry of the action expert. ``None`` keeps the fused
        # ``qkv = Linear(dim, dim*3)`` MHA layout used by every legacy
        # checkpoint; both must be set together for the prefix-KV expert.
        num_kv_heads: int | None = None,
        attn_head_dim: int | None = None,
        # Action-condition source. Train may use GT, generated or a mixture;
        # validation/eval always force generated and never read GT captions.
        pi05_condition_source: str = "ground_truth",
        pi05_condition_generated_prob: float = 0.0,
        # Load-only tombstone: True is rejected. Figure 18 lives inside
        # vlm_fast_action_bridge and is not a combinable flag.
        pi05_joint_attention_mask: bool = False,
        # Versioned action-policy handoff contract. ``auto`` resolves from the
        # selected backend at model construction; raw checkpoint validators
        # still require an explicit trained contract for deployment/evaluation.
        vlm_policy_contract: str = "auto",
        vlm_policy_vision_contract: dict | None = None,
        vlm_policy_semantic_layers: list[int] | tuple[int, ...] | None = None,
        # ``readonly_cross_attention_v2`` only: which semantic depth each
        # cross-attn layer reads, as indices into vlm_policy_semantic_layers.
        # ``None`` spreads the depths monotonically over the cross-attn layers
        # so the first one reads the shallowest and the last one the final.
        action_context_semantic_route: list[int] | tuple[int, ...] | None = None,
        # Action-policy text cap, independent from the VLM's native context
        # window. Serialize it so PT, finetune, and deploy share one contract.
        action_language_max_length: int = 50,
        action_initialization_contract: str = "legacy_or_pretrained",
        action_initialization_seed: int | None = None,
        policy_architecture_source: str = "",

        # === Action & proprio ===
        max_action_dim: int = 20,  # Maximum action dimension for padding (used by "auto" action mode)
        real_action_dim: int = 20,
        num_actions: int = 30,
        action_mode: str = "ee6d",
        action_loss_mode: str = "regression",
        action_loss_profile: str = "hy_normalized_mse",
        ee6d_regression_normalization: bool = False,
        action_norm_stats_filename: str = "",
        action_norm_stats_sha256: str = "",
        use_proprio: bool = True,
        cross_emb_ee_equivalent: bool = False,
        # Number of per-dataset action norm banks carried by the cross_emb
        # action space. The banks are runtime state derived from
        # --action_norm_stats, but their row count decides buffer shapes, so it
        # must be serialized: otherwise from_pretrained rebuilds 1-row buffers
        # and every multi-bank checkpoint trips the shape check on load.
        action_norm_bank_count: int = 1,
        keep_vlm_decoder: bool = False,
        ki_detach_vlm_to_expert: bool = False,
        gripper_loss_type: str = "bce",
        keep_gripper_proprio: bool = False,

        # === propri/FAST cotraining vocab layout (self-describing) ===
        # Offsets into the merged Florence-2-base-ac-propri vocab. None means
        # "this is not a propri-extended checkpoint" (e.g. action-only Arm A) —
        # at runtime they fall back to PROPRI_VOCAB_LAYOUT canonical constants
        # via models.propri_vocab.resolve_propri_offsets. When a cotrain run
        # writes them here, the produced checkpoint is self-describing.
        action_token_start_index: int | None = None,
        propri_token_start_index: int | None = None,
        merged_vocab_size: int | None = None,

        # === UAM Dorsal (future-observation visual dynamics) ===
        # Independent predictor emits dedicated visual-delta tokens. The action
        # expert routes semantic and Dorsal streams without overwriting VLM tokens.
        dorsal_enabled: bool = False,
        dorsal_feat_dim: int = 1024,
        dorsal_hidden_dim: int = 1024,
        dorsal_depth: int = 6,
        dorsal_num_heads: int = 16,
        dorsal_loss_weight: float = 0.5,
        dorsal_align_loss_weight: float = 0.25,
        dorsal_dynamics_loss_weight: float = 1.0,
        dorsal_delta_direction_loss_weight: float = 0.5,
        dorsal_delta_scale_loss_weight: float = 0.1,
        dorsal_gate_max: float = 1.0,
        dorsal_gate_reg_weight: float = 1e-4,
        dorsal_target_mode: str = "next",
        dorsal_num_spatial_tokens: int = 16,
        dorsal_feature_preserving_init: bool = True,
        dorsal_input_source: str = "independent_davit_multiview",
        dorsal_input_adapter_trained: bool = False,
        dorsal_vision_encoder_trained: bool = False,
        dorsal_view_mode: str = "all_valid_views",
        dorsal_delta_supervision: str = "teacher_feature_delta_v1",
        dorsal_target_provider: str = "dinov3_vitl16_frozen_training_only",
        dorsal_injection_mode: str = "separate_tokens_v1",
        dorsal_router_init_prob: float = 0.1,
        dorsal_router_trained: bool = False,
        dorsal_router_attention: str = "separate_cross_attention_v1",
        dorsal_stream_identity: str = "learned_type_embedding",
        dorsal_history_frames: int = 1,
        dorsal_history_span_sec: float = 0.0,
        dorsal_temporal_input: str = "current_only",
        dorsal_temporal_fusion_trained: bool = False,
        dorsal_history_encoder_mode: str = "full_davit_v1",
        dorsal_history_shallow_stages: int = 2,
        dorsal_history_motion_encoder_trained: bool = False,
        dorsal_vision_gradient_checkpointing: bool = False,
        dorsal_inference_cache_size: int = 128,
        # Deprecated residual-primary fields retained only for old config loading.
        dorsal_gate_trained: bool = False,

        # === DINOv3 latent target supervision (legacy; prefer dorsal_enabled) ===
        dino_enabled: bool = False,
        dino_target_dim: int = 1024,
        dino_num_latent_tokens: int = 4,
        dino_loss_weight: float = 0.5,
        dino_weights_path: str = "",
        # Legacy DINO forward-prediction horizon: 0 aligns the current-frame CLS;
        # horizon >= 1 predicts f_{t+horizon}. UAM Dorsal uses a separate
        # future-observation spatial-token path, so this setting remains for
        # legacy latent/DINO ablations and Dorsal ``next`` target selection.
        dino_horizon: int = 0,
        # latent: auxiliary on action-expert latent tokens (legacy).
        # davit: distill DaViT pooled visual tokens toward DINO CLS (no latent tokens).
        dino_supervision_site: str = "latent",
        # When True, action/visual tokens cannot attend to latent prefix tokens.
        dino_detach_action: bool = False,
        dino_loss_type: str = "mse",
        # When True, latent prefix tokens use pos_emb[0:offset] while action/visual
        # streams keep pos_emb[0:rest] (no shift). Required for fresh DINO pretrain
        # from a non-dino checkpoint; legacy coupled ckpts leave this False.
        dino_pos_emb_decoupled: bool = False,

        **kwargs,
    ):
        # Florence2 backbone configuration
        if isinstance(florence_config, dict):
            self.florence_config = Florence2Config(**florence_config)
        elif isinstance(florence_config, Florence2Config):
            self.florence_config = florence_config
        else:
            self.florence_config = Florence2Config()
        self.vlm_backend = str(vlm_backend or "florence2").strip().lower()
        self.vlm_config = dict(vlm_config) if isinstance(vlm_config, dict) else None
        self.vlm_model_name_or_path = str(vlm_model_name_or_path or "")
        self.vlm_projection_dim = int(vlm_projection_dim or 0)
        self.vlm_dtype = str(vlm_dtype or "auto").strip().lower()
        self.vlm_lora = vlm_lora
        self.vlm_max_view_embeddings = int(vlm_max_view_embeddings)
        self.qwen_temporal_embedding = str(
            qwen_temporal_embedding or "off"
        ).strip().lower()
        self.qwen_image_attention_mode = str(qwen_image_attention_mode).strip().lower()
        self.qwen_far_aux = dict(qwen_far_aux) if qwen_far_aux is not None else None
        self.qwen_history_time_rope = qwen_history_time_rope
        if self.qwen_temporal_embedding not in {
            "off",
            "fixed_sinusoidal_v1",
            "spacetime_mem_v1",
            "vlm_history_current_v1",
            "vlm_history_value_v1",
        }:
            raise ValueError(
                "qwen_temporal_embedding must be one of "
                "off / fixed_sinusoidal_v1 / spacetime_mem_v1 / vlm_history_current_v1 / vlm_history_value_v1, got "
                f"{self.qwen_temporal_embedding!r}"
            )
        self.qwen_spacetime_stride = int(qwen_spacetime_stride)
        self.qwen_history_value_interval = int(qwen_history_value_interval)
        if self.qwen_history_value_interval < 1:
            raise ValueError("qwen_history_value_interval must be positive.")
        self.qwen_history_value_rope_theta = float(qwen_history_value_rope_theta)
        if self.qwen_history_value_rope_theta <= 1.0:
            raise ValueError("qwen_history_value_rope_theta must exceed 1.")
        if self.qwen_spacetime_stride < 1:
            raise ValueError(
                "qwen_spacetime_stride must be >= 1, got "
                f"{self.qwen_spacetime_stride}"
            )
        self.qwen_history_memory_enabled = bool(qwen_history_memory_enabled)
        self.qwen_history_memory_pool_size = int(qwen_history_memory_pool_size)
        if self.qwen_history_memory_pool_size < 1:
            raise ValueError(
                "qwen_history_memory_pool_size must be >= 1, got "
                f"{self.qwen_history_memory_pool_size}"
            )
        self.qwen_history_memory_gate_init = float(
            qwen_history_memory_gate_init
        )
        if not 0.0 <= self.qwen_history_memory_gate_init < 1.0:
            raise ValueError(
                "qwen_history_memory_gate_init must be in [0,1), got "
                f"{self.qwen_history_memory_gate_init}"
            )
        self.qwen_native_video_history_enabled = bool(
            qwen_native_video_history_enabled
        )
        if self.qwen_temporal_embedding in {"vlm_history_current_v1", "vlm_history_value_v1"} and (
            self.qwen_native_video_history_enabled or self.qwen_history_memory_enabled
        ):
            raise ValueError(
                f"{self.qwen_temporal_embedding} cannot be combined with P3/history bank."
            )
        if qwen_native_video_history_views is None:
            self.qwen_native_video_history_views: tuple[int, ...] = ()
        else:
            views = tuple(int(v) for v in qwen_native_video_history_views)
            if any(v < 0 for v in views):
                raise ValueError(
                    "qwen_native_video_history_views entries must be >= 0, "
                    f"got {views}"
                )
            if len(set(views)) != len(views):
                raise ValueError(
                    "qwen_native_video_history_views entries must be unique, "
                    f"got {views}"
                )
            self.qwen_native_video_history_views = views
        if (
            self.qwen_native_video_history_enabled
            and not self.qwen_native_video_history_views
        ):
            raise ValueError(
                "qwen_native_video_history_enabled=True requires at least "
                "one view index in qwen_native_video_history_views"
            )
        self.qwen_action_adalayernorm_enabled = bool(
            qwen_action_adalayernorm_enabled
        )
        self.dm05_action_config = (
            dict(dm05_action_config) if isinstance(dm05_action_config, dict) else None
        )
        if self.vlm_max_view_embeddings <= 0:
            raise ValueError(
                "vlm_max_view_embeddings must be positive, got "
                f"{self.vlm_max_view_embeddings}"
            )
        self.vlm_prompt_config = (
            dict(vlm_prompt_config)
            if isinstance(vlm_prompt_config, dict)
            else {}
        )
        if self.vlm_backend in {"florence", "florence2"}:
            self.vlm_backend = "florence2"
            if self.vlm_projection_dim <= 0:
                self.vlm_projection_dim = int(self.florence_config.projection_dim)
        elif self.vlm_backend in {"qwen3.5", "qwen3.5-2b", "qwen3.5-4b", "qwen35"}:
            self.vlm_backend = "qwen3_5"
        elif self.vlm_backend in {"dm05", "dm0.5", "dm_05"}:
            self.vlm_backend = "dm05"
        if self.vlm_projection_dim <= 0:
            raise ValueError(
                "vlm_projection_dim must be positive for non-Florence backends."
            )

        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_domains = num_domains
        self.len_soft_prompts = len_soft_prompts
        self.dim_time = dim_time
        self.time_embedding_max_period = int(time_embedding_max_period)
        self.time_embedding_input_scale = float(time_embedding_input_scale)
        flow_time_distribution_norm = str(flow_time_distribution or "beta").strip().lower()
        if flow_time_distribution_norm not in {"beta", "logit_normal"}:
            raise ValueError(
                "flow_time_distribution must be 'beta' or 'logit_normal'; "
                f"got {flow_time_distribution!r}"
            )
        self.flow_time_distribution = flow_time_distribution_norm
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)
        if self.beta_a <= 0.0 or self.beta_b <= 0.0:
            raise ValueError(
                "beta_a and beta_b must be strictly positive; got "
                f"beta_a={beta_a!r}, beta_b={beta_b!r}"
            )
        self.logit_normal_m = float(logit_normal_m)
        self.logit_normal_s = float(logit_normal_s)
        if self.flow_time_distribution == "logit_normal" and self.logit_normal_s <= 0.0:
            raise ValueError(
                "logit_normal_s must be strictly positive for the logit_normal "
                f"schedule; got {self.logit_normal_s!r}"
            )
        self.max_len_seq = max_len_seq
        self.use_hetero_proj = use_hetero_proj
        self.soft_prompt_length = soft_prompt_length
        self.action_context_mode = canonicalize_action_context_mode(action_context_mode)
        if self.action_context_mode not in {
            "auto",
            "legacy_joint_v1",
            "readonly_cross_attention_v1",
            "readonly_cross_attention_v2",
            "prefix_kv",
        }:
            raise ValueError(
                "action_context_mode must be auto, legacy_joint_v1, "
                "readonly_cross_attention_v1, readonly_cross_attention_v2, or prefix_kv; "
                f"got {self.action_context_mode!r}"
            )
        self.action_context_cross_attn_every_n_layers = int(
            action_context_cross_attn_every_n_layers
        )
        if self.action_context_cross_attn_every_n_layers <= 0:
            raise ValueError(
                "action_context_cross_attn_every_n_layers must be positive, "
                f"got {self.action_context_cross_attn_every_n_layers}"
            )
        self.action_context_cross_attn_gate_init = float(
            action_context_cross_attn_gate_init
        )
        if not 0.0 < self.action_context_cross_attn_gate_init < 1.0:
            raise ValueError(
                "action_context_cross_attn_gate_init must be in (0,1), got "
                f"{self.action_context_cross_attn_gate_init}"
            )
        self.action_context_trained = bool(action_context_trained)
        # --- vlm_fast_action_bridge (G0.5 prefix-KV) contract ---------------
        raw_architecture_version = str(
            architecture_version or LEGACY_ARCHITECTURE_VERSION
        ).strip().lower()
        if raw_architecture_version not in ALLOWED_ARCHITECTURE_VERSIONS:
            raise ValueError(
                "architecture_version must be one of "
                f"{sorted(ALLOWED_ARCHITECTURE_VERSIONS)}, got "
                f"{raw_architecture_version!r}."
            )
        self.architecture_version = canonicalize_architecture_version(
            raw_architecture_version
        )
        reject_removed_suffix_adapter(
            {
                "architecture_version": self.architecture_version,
                "pi05_joint_attention_mask": pi05_joint_attention_mask
                or kwargs.get("pi05_joint_attention_mask", False),
            }
        )
        self.num_kv_heads = None if num_kv_heads is None else int(num_kv_heads)
        self.attn_head_dim = (
            None if attn_head_dim is None else int(attn_head_dim)
        )
        if (self.num_kv_heads is None) != (self.attn_head_dim is None):
            raise ValueError(
                "num_kv_heads and attn_head_dim must be set together; omit "
                "both to keep the fused MHA action expert."
            )
        self.condition_state_version = str(
            condition_state_version or "none"
        ).strip()
        self.prefix_layout_version = canonicalize_prefix_layout_version(
            prefix_layout_version or "legacy_v5"
        )
        if not self.condition_state_version or not self.prefix_layout_version:
            raise ValueError(
                "condition_state_version and prefix_layout_version must be "
                "non-empty."
            )
        self.semantic_anchor_policy = canonicalize_semantic_anchor_policy(
            semantic_anchor_policy
        )
        self.fast_visibility_policy = canonicalize_fast_visibility_policy(
            fast_visibility_policy or "legacy_semantic_capture_v1"
        )
        self.matched_dual_stream_fast_ce_version = str(
            matched_dual_stream_fast_ce_version or "none"
        ).strip().lower()
        if self.matched_dual_stream_fast_ce_version not in {
            "none",
            "shifted_native_v1",
        }:
            raise ValueError(
                "matched_dual_stream_fast_ce_version must be none or "
                "shifted_native_v1, got "
                f"{self.matched_dual_stream_fast_ce_version!r}."
            )
        self.matched_dual_stream_route = (
            None
            if matched_dual_stream_route is None
            else [int(index) for index in matched_dual_stream_route]
        )
        if self.matched_dual_stream_route is not None and any(
            index < 0 for index in self.matched_dual_stream_route
        ):
            raise ValueError(
                "matched_dual_stream_route indices must be non-negative, got "
                f"{self.matched_dual_stream_route}."
            )
        if is_vlm_fast_action_bridge_version(self.architecture_version):
            # Hy-Embodied only implements the prefix-KV member of this family.
            # The FAST-in-expert `matched_dual_stream_v1` bridge is not ported,
            # so refuse it loudly instead of silently building a different net.
            if self.num_kv_heads is None:
                raise ValueError(
                    "Hy-Embodied supports vlm_fast_action_bridge only through "
                    "the prefix_kv expert; the checkpoint must declare "
                    "num_kv_heads and attn_head_dim."
                )
            if self.matched_dual_stream_fast_ce_version != "none":
                raise ValueError(
                    "vlm_fast_action_bridge keeps FAST in the VLM AR path; "
                    "matched_dual_stream_fast_ce_version must be 'none'."
                )
            if self.attn_head_dim <= 0:
                raise ValueError(
                    "vlm_fast_action_bridge prefix_kv requires a positive "
                    "attn_head_dim matching the VLM."
                )
            if (
                not is_prefix_kv_action_context_mode(self.action_context_mode)
                and self.action_context_mode != "auto"
            ):
                raise ValueError(
                    "vlm_fast_action_bridge with GQA requires "
                    "action_context_mode='prefix_kv'."
                )
            self.action_context_mode = "prefix_kv"
        elif is_prefix_kv_action_context_mode(self.action_context_mode):
            raise ValueError(
                "action_context_mode='prefix_kv' requires "
                "architecture_version='vlm_fast_action_bridge'."
            )
        self.pi05_condition_source = normalize_condition_source(
            pi05_condition_source
        )
        self.pi05_condition_generated_prob = float(pi05_condition_generated_prob)
        if not 0.0 <= self.pi05_condition_generated_prob <= 1.0:
            raise ValueError(
                "pi05_condition_generated_prob must be in [0, 1], got "
                f"{self.pi05_condition_generated_prob}."
            )
        if self.pi05_condition_source == "scheduled_mixture" and not (
            0.0 < self.pi05_condition_generated_prob < 1.0
        ):
            raise ValueError(
                "scheduled_mixture requires 0 < pi05_condition_generated_prob < 1."
            )
        if self.pi05_condition_source == "generated":
            self.pi05_condition_generated_prob = 1.0
        if self.pi05_condition_source == "ground_truth":
            self.pi05_condition_generated_prob = 0.0
        _resolved_path = resolve_architecture_path(self)
        self.architecture_version = _resolved_path["architecture_version"]
        self.architecture_path = _resolved_path["architecture_path"]
        self.cache_topology = _resolved_path["cache_topology"]
        self.fast_visibility_policy = _resolved_path["fast_visibility_policy"]
        # ``architecture_path``/``cache_topology`` are accepted as constructor
        # arguments so a serialized config round-trips, but they are always
        # re-derived above: a stale value on disk can never disagree with the
        # resolver, and because they are named parameters they never reach
        # ``kwargs`` and cannot be re-serialized unresolved.
        self.vlm_policy_contract = str(
            vlm_policy_contract or "auto"
        ).strip().lower()
        if not self.vlm_policy_contract:
            raise ValueError("vlm_policy_contract must be non-empty.")
        if vlm_policy_vision_contract is None:
            self.vlm_policy_vision_contract = None
        elif isinstance(vlm_policy_vision_contract, dict):
            self.vlm_policy_vision_contract = dict(vlm_policy_vision_contract)
        else:
            raise ValueError(
                "vlm_policy_vision_contract must be a dict or None, got "
                f"{type(vlm_policy_vision_contract).__name__}."
            )
        self.vlm_policy_semantic_layers = (
            None
            if vlm_policy_semantic_layers is None
            else [int(index) for index in vlm_policy_semantic_layers]
        )
        if self.vlm_policy_semantic_layers is not None:
            if not self.vlm_policy_semantic_layers:
                raise ValueError(
                    "vlm_policy_semantic_layers cannot be an empty list."
                )
            if any(index < 0 for index in self.vlm_policy_semantic_layers):
                raise ValueError(
                    "vlm_policy_semantic_layers must contain non-negative "
                    f"indices, got {self.vlm_policy_semantic_layers}."
                )
            if len(set(self.vlm_policy_semantic_layers)) != len(
                self.vlm_policy_semantic_layers
            ):
                raise ValueError(
                    "vlm_policy_semantic_layers must not contain duplicates, "
                    f"got {self.vlm_policy_semantic_layers}."
                )
        self.action_context_semantic_route = (
            None
            if action_context_semantic_route is None
            else [int(index) for index in action_context_semantic_route]
        )
        if self.action_context_semantic_route is not None:
            if self.action_context_mode != "readonly_cross_attention_v2":
                raise ValueError(
                    "action_context_semantic_route is only valid with "
                    "action_context_mode=readonly_cross_attention_v2."
                )
            if not self.action_context_semantic_route:
                raise ValueError(
                    "action_context_semantic_route cannot be an empty list."
                )
            expected_route_length = len(
                range(
                    0,
                    self.depth,
                    self.action_context_cross_attn_every_n_layers,
                )
            )
            if len(self.action_context_semantic_route) != expected_route_length:
                raise ValueError(
                    "action_context_semantic_route needs one entry per "
                    f"cross-attn layer ({expected_route_length}), got "
                    f"{len(self.action_context_semantic_route)}."
                )
            if any(index < 0 for index in self.action_context_semantic_route):
                raise ValueError(
                    "action_context_semantic_route must contain non-negative "
                    f"indices, got {self.action_context_semantic_route}."
                )
            if self.vlm_policy_semantic_layers is not None and any(
                index >= len(self.vlm_policy_semantic_layers)
                for index in self.action_context_semantic_route
            ):
                raise ValueError(
                    "action_context_semantic_route indices must be < "
                    f"{len(self.vlm_policy_semantic_layers)} "
                    "(len(vlm_policy_semantic_layers)), got "
                    f"{self.action_context_semantic_route}."
                )
        self.action_language_max_length = int(action_language_max_length)
        if self.action_language_max_length <= 0:
            raise ValueError(
                "action_language_max_length must be positive, got "
                f"{self.action_language_max_length}."
            )
        self.action_initialization_contract = str(
            action_initialization_contract or "legacy_or_pretrained"
        ).strip().lower()
        self.action_initialization_seed = (
            None
            if action_initialization_seed is None
            else int(action_initialization_seed)
        )
        self.policy_architecture_source = str(
            policy_architecture_source or ""
        )

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.action_loss_mode = action_loss_mode
        self.action_loss_profile = str(action_loss_profile)
        self.ee6d_regression_normalization = bool(ee6d_regression_normalization)
        self.action_norm_stats_filename = str(action_norm_stats_filename)
        self.action_norm_stats_sha256 = str(action_norm_stats_sha256)
        self.use_proprio = use_proprio
        self.cross_emb_ee_equivalent = bool(cross_emb_ee_equivalent)
        self.action_norm_bank_count = max(1, int(action_norm_bank_count))
        self.keep_vlm_decoder = bool(keep_vlm_decoder)
        # π0.5-KI: detach VLM features before they enter the action expert so
        # the FM action loss only backprops into expert/proprio/visual proj.
        # VLM CE loss (FAST + caption) still flows through the VLM normally.
        self.ki_detach_vlm_to_expert = bool(ki_detach_vlm_to_expert)
        gripper_loss_type_norm = str(gripper_loss_type).strip().lower()
        if gripper_loss_type_norm not in ("bce", "mse"):
            raise ValueError(
                f"gripper_loss_type must be 'bce' or 'mse', got {gripper_loss_type!r}"
            )
        self.gripper_loss_type = gripper_loss_type_norm
        self.keep_gripper_proprio = bool(keep_gripper_proprio)

        self.real_action_dim = real_action_dim
        self.max_action_dim = max_action_dim
        self.dm05_state_dim = int(dm05_state_dim)
        if self.dm05_state_dim <= 0 or self.dm05_state_dim > self.max_action_dim:
            raise ValueError(
                "dm05_state_dim must be in [1, max_action_dim], got "
                f"{self.dm05_state_dim} with max_action_dim={self.max_action_dim}"
            )

        # propri/FAST vocab offsets — kept as-is (None when absent) so the
        # checkpoint round-trips its own layout; resolution to canonical
        # defaults happens at use sites, not here.
        self.action_token_start_index = (
            None if action_token_start_index is None else int(action_token_start_index)
        )
        self.propri_token_start_index = (
            None if propri_token_start_index is None else int(propri_token_start_index)
        )
        self.merged_vocab_size = (
            None if merged_vocab_size is None else int(merged_vocab_size)
        )

        # Drop fields written by the superseded inference-time DINO contract.
        # Otherwise PretrainedConfig would preserve and re-serialize them.
        kwargs.pop("dorsal_feature_provider", None)
        kwargs.pop("dorsal_preprocess_contract", None)

        # UAM Dorsal
        self.dorsal_enabled = bool(dorsal_enabled)
        self.dorsal_feat_dim = int(dorsal_feat_dim)
        self.dorsal_hidden_dim = int(dorsal_hidden_dim)
        self.dorsal_depth = int(dorsal_depth)
        self.dorsal_num_heads = int(dorsal_num_heads)
        self.dorsal_loss_weight = float(dorsal_loss_weight)
        self.dorsal_align_loss_weight = float(dorsal_align_loss_weight)
        self.dorsal_dynamics_loss_weight = float(dorsal_dynamics_loss_weight)
        self.dorsal_delta_direction_loss_weight = float(
            dorsal_delta_direction_loss_weight
        )
        self.dorsal_delta_scale_loss_weight = float(
            dorsal_delta_scale_loss_weight
        )
        self.dorsal_gate_max = float(dorsal_gate_max)
        self.dorsal_gate_reg_weight = float(dorsal_gate_reg_weight)
        self.dorsal_target_mode = str(dorsal_target_mode or "next").strip().lower()
        if self.dorsal_target_mode == "goal":
            self.dorsal_target_mode = "episode_end"
        if self.dorsal_target_mode not in {
            "next", "action_step", "chunk_end", "episode_end"
        }:
            raise ValueError(
                "dorsal_target_mode must be next, action_step, chunk_end, "
                "or episode_end; "
                f"got {self.dorsal_target_mode!r}"
            )
        self.dorsal_num_spatial_tokens = int(dorsal_num_spatial_tokens)
        self.dorsal_feature_preserving_init = bool(dorsal_feature_preserving_init)
        self.dorsal_input_source = str(
            dorsal_input_source or "independent_davit_multiview"
        )
        self.dorsal_input_adapter_trained = bool(dorsal_input_adapter_trained)
        self.dorsal_vision_encoder_trained = bool(dorsal_vision_encoder_trained)
        self.dorsal_view_mode = str(
            dorsal_view_mode or "all_valid_views"
        ).strip().lower()
        if self.dorsal_view_mode != "all_valid_views":
            raise ValueError(
                "dorsal_view_mode must be 'all_valid_views', got "
                f"{self.dorsal_view_mode!r}"
            )
        self.dorsal_delta_supervision = str(
            dorsal_delta_supervision or "teacher_feature_delta_v1"
        ).strip().lower()
        if self.dorsal_delta_supervision != "teacher_feature_delta_v1":
            raise ValueError(
                "dorsal_delta_supervision must be "
                f"'teacher_feature_delta_v1', got {self.dorsal_delta_supervision!r}"
            )
        self.dorsal_target_provider = str(
            dorsal_target_provider or "dinov3_vitl16_frozen_training_only"
        )
        self.dorsal_injection_mode = str(
            dorsal_injection_mode or "separate_tokens_v1"
        )
        if self.dorsal_injection_mode not in {
            "separate_tokens_v1",
            "residual_primary",
        }:
            raise ValueError(
                "dorsal_injection_mode must be separate_tokens_v1 or "
                f"residual_primary; got {self.dorsal_injection_mode!r}"
            )
        self.dorsal_router_init_prob = float(dorsal_router_init_prob)
        if not 0.0 < self.dorsal_router_init_prob < 1.0:
            raise ValueError(
                "dorsal_router_init_prob must be in (0,1), got "
                f"{self.dorsal_router_init_prob}"
            )
        self.dorsal_router_trained = bool(dorsal_router_trained)
        self.dorsal_router_attention = str(
            dorsal_router_attention or "separate_cross_attention_v1"
        ).strip().lower()
        if self.dorsal_router_attention != "separate_cross_attention_v1":
            raise ValueError(
                "dorsal_router_attention must be "
                f"'separate_cross_attention_v1', got {self.dorsal_router_attention!r}"
            )
        self.dorsal_stream_identity = str(
            dorsal_stream_identity or "learned_type_embedding"
        ).strip().lower()
        if self.dorsal_stream_identity != "learned_type_embedding":
            raise ValueError(
                "dorsal_stream_identity must be 'learned_type_embedding', got "
                f"{self.dorsal_stream_identity!r}"
            )
        self.dorsal_history_frames = int(dorsal_history_frames)
        self.dorsal_history_span_sec = float(dorsal_history_span_sec)
        if self.dorsal_history_frames <= 0:
            raise ValueError(
                "dorsal_history_frames must be positive, got "
                f"{self.dorsal_history_frames}"
            )
        if (
            self.dorsal_history_frames > 1
            and self.dorsal_history_span_sec <= 0.0
        ):
            raise ValueError(
                "dorsal_history_span_sec must be positive when "
                f"dorsal_history_frames={self.dorsal_history_frames}"
            )
        self.dorsal_temporal_input = str(
            dorsal_temporal_input
            or (
                "causal_short_history_delta_v1"
                if self.dorsal_history_frames > 1
                else "current_only"
            )
        ).strip().lower()
        expected_temporal_input = (
            "causal_short_history_delta_v1"
            if self.dorsal_history_frames > 1
            else "current_only"
        )
        if self.dorsal_temporal_input != expected_temporal_input:
            raise ValueError(
                "dorsal_temporal_input does not match dorsal_history_frames: "
                f"expected {expected_temporal_input!r}, got "
                f"{self.dorsal_temporal_input!r}"
            )
        self.dorsal_temporal_fusion_trained = bool(
            dorsal_temporal_fusion_trained
        )
        self.dorsal_history_encoder_mode = str(
            dorsal_history_encoder_mode or "full_davit_v1"
        ).strip().lower()
        if self.dorsal_history_encoder_mode not in {
            "full_davit_v1",
            "asymmetric_shallow_v1",
        }:
            raise ValueError(
                "dorsal_history_encoder_mode must be full_davit_v1 or "
                f"asymmetric_shallow_v1, got {self.dorsal_history_encoder_mode!r}"
            )
        self.dorsal_history_shallow_stages = int(
            dorsal_history_shallow_stages
        )
        if self.dorsal_history_encoder_mode == "asymmetric_shallow_v1":
            num_vision_stages = len(self.florence_config.vision_config.dim_embed)
            if not 1 <= self.dorsal_history_shallow_stages < num_vision_stages:
                raise ValueError(
                    "dorsal_history_shallow_stages must be in "
                    f"[1,{num_vision_stages - 1}], got "
                    f"{self.dorsal_history_shallow_stages}"
                )
        self.dorsal_history_motion_encoder_trained = bool(
            dorsal_history_motion_encoder_trained
        )
        self.dorsal_vision_gradient_checkpointing = bool(
            dorsal_vision_gradient_checkpointing
        )
        self.dorsal_inference_cache_size = int(dorsal_inference_cache_size)
        if self.dorsal_inference_cache_size < 0:
            raise ValueError(
                "dorsal_inference_cache_size must be non-negative, got "
                f"{self.dorsal_inference_cache_size}"
            )
        self.dorsal_gate_trained = bool(dorsal_gate_trained)

        # DINOv3 latent target (legacy)
        self.dino_enabled = bool(dino_enabled)
        self.dino_target_dim = int(dino_target_dim)
        self.dino_num_latent_tokens = int(dino_num_latent_tokens)
        self.dino_loss_weight = float(dino_loss_weight)
        self.dino_weights_path = str(dino_weights_path or "")
        self.dino_horizon = int(dino_horizon)
        self.dino_supervision_site = str(dino_supervision_site or "latent").strip().lower()
        self.dino_detach_action = bool(dino_detach_action)
        self.dino_loss_type = str(dino_loss_type or "mse").strip().lower()
        self.dino_pos_emb_decoupled = bool(dino_pos_emb_decoupled)

        # Initialize base HF config attributes (e.g. name_or_path)
        super().__init__(**kwargs)
        from .qwen_image_attention import validate_image_attention

        validate_image_attention(self)
        if self.qwen_far_aux and self.qwen_far_aux.get("enabled", False):
            raise ValueError("Era0-Preview evaluation does not support auxiliary heads.")
        from .qwen_history_time_rope import validate_config as validate_time_rope
        validate_time_rope(self)

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert this configuration (and its Florence sub-config)
        into a fully serializable dictionary for HF save/load.
        """
        output = super().to_dict()
        output["florence_config"] = self.florence_config.to_dict()
        output["vlm_config"] = self.vlm_config
        output["dm05_action_config"] = self.dm05_action_config
        return output
