import argparse

import torch
from transformers import AutoModelForCausalLM

from a1 import Molmo, ModelConfig, ActivationType, LayerNormType, TokenizerConfig, \
    VisionBackboneConfig
from a1.config import ImagePooling2DType


def load_hf_model(repo_id) -> Molmo:
    if repo_id == "allenai/Molmo-7B-D-0924":
        tokenizer = 'allenai/OLMoE-1B-7B-0924'
    else:
        raise NotImplementedError(repo_id)

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True, torch_dtype=torch.float32)

    cfg = model.config
    model_cfg = ModelConfig(
        d_model=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads,
        qkv_bias=cfg.qkv_bias,
        clip_qkv=cfg.clip_qkv,
        n_layers=cfg.num_hidden_layers,
        mlp_hidden_size=cfg.intermediate_size,
        activation_type=ActivationType.swiglu,
        block_type="sequential",
        attention_layer_norm=cfg.attention_layer_norm,
        residual_dropout=0,
        response_residual_dropout=0,
        embedding_dropout=0,
        layer_norm_type=LayerNormType.rms,
        layer_norm_with_affine=True,
        layer_norm_eps=cfg.layer_norm_eps,
        attention_layer_norm_with_affine=True,
        max_position_embeddings=cfg.max_position_embeddings,
        include_bias=False,
        bias_for_layer_norm=False,
        scale_logits=False,
        vocab_size=cfg.vocab_size,
        embedding_size=cfg.embedding_size,
        additional_vocab_size=128,
        new_embedding_init_range=0.02,
        weight_tying=False,
        norm_after=cfg.norm_after,
        max_crops=12,
        crop_mode="overlap-and-resize-c2",
        use_col_tokens=True,
        prompt_type="uber_model",
        message_formatting="role",
        system_prompt_kind="demo_or_style",
        multi_annotation_weighting="root_subsegments",
        image_padding_embed="pad_and_partial_pad",
        fix_image_padding=False,
        rope=True,
        rope_theta=cfg.rope_theta,
        vit_layers=[-2, -9],
        image_pooling_2d=ImagePooling2DType.attention_meanq,
        normalize_input_embeds=False,
        tokenizer=TokenizerConfig(identifier=tokenizer),
        vision_backbone=VisionBackboneConfig(
            image_model_type="openai",
            image_default_input_size=(336, 336),
            image_patch_size=14,
            image_pos_patch_size=14,
            image_emb_dim=1024,
            image_num_heads=16,
            image_num_key_value_heads=16,
            image_num_layers=23,
            image_head_dim=64,
            image_mlp_dim=4096,
            image_mlp_activations="quick_gelu",
            image_dropout_rate=0.0,
            image_num_pos=577,
            image_norm_eps=1e-5,
            attention_dropout=0.0,
            residual_dropout=0.0,
            initializer_range=0.02,
        )
    )

    olmo_model = Molmo(model_cfg, init_params=False)
    state_dict = model.state_dict()
    state_dict = {k[6:]: v for k, v in state_dict.items()}
    olmo_model.load_state_dict(state_dict)


def convert_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    load_hf_model(args.repo_id)


if __name__ == '__main__':
    load_hf_model("allenai/Molmo-7B-D-0924")