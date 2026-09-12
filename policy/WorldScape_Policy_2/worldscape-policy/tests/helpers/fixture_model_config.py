"""Minimal native model configs for unit tests."""

from __future__ import annotations


def fixture_native_model_config(*, auto: bool = False) -> dict:
    return {
        "schema_version": "1",
        "model": {
            "mode": "auto" if auto else "interactive",
            "shape": {
                "num_frames": 3,
                "frame_block_size": 2,
                "actions_per_block": 24,
                "states_per_block": 1,
                "action_horizon": 24,
                "action_dim": 3,
                "max_state_dim": 3,
                "vlm_token_dim": 4,
                "condition_dim": 4,
            },
            "condition_router": {
                "auto": {
                    "vlm": {"target": "fixture.Qwen", "parameters": {}},
                    "projector": {
                        "kind": "linear",
                        "input_dim": 4,
                        "output_dim": 4,
                    },
                    "output_norm": auto,
                    "semantic_gate_only": True,
                    "semantic_grad_clip_norm": 0.5,
                },
                "interactive": {
                    "t5": {
                        "target": "fixture.T5",
                        "parameters": {"tokenizer_path": "org/test-tokenizer"},
                    }
                },
            },
            "event_memory": {
                "enabled": False,
                "history_steps": 8,
                "global_slots": 1,
                "local_steps": 4,
                "boundary_steps": 8,
                "boundary_min_gap": 1,
                "perception_gist_tokens": 8,
                "residual_scale": 0.1,
                "dropout": 0.0,
            },
            "visual_memory": {
                "vae": {"target": "fixture.VAE", "parameters": {}},
                "image_encoder": {"target": "fixture.Image", "parameters": {}},
                "persistent_prompt": "goal_or_demo",
                "diffusion_view_layout": "mosaic_2x2",
                "view_index": 0,
                "tiled": False,
                "tile_size": [34, 34],
                "tile_stride": [18, 16],
            },
            "wam": {
                "plugin": "wan22",
                "variant": "ti2v-5b",
                "core": {"target": "fixture.CausalWanModel", "parameters": {}},
                "num_timestep_buckets": 1000,
                "train_architecture": "full",
                "decouple_inference_noise": False,
                "video_inference_final_noise": 0.8,
                "decouple_video_action_noise": False,
                "video_noise_beta_alpha": 3.0,
                "video_noise_beta_beta": 1.0,
                "use_high_noise_emphasis": False,
                "high_noise_beta_alpha": 3.0,
                "high_noise_beta_beta": 1.0,
            },
        },
    }
