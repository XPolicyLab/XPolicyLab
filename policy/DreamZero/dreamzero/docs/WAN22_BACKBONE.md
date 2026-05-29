# Training DreamZero with Wan2.2-TI2V-5B Backbone

This guide explains how to train DreamZero on the DROID dataset using **Wan2.2-TI2V-5B** as the backbone instead of the default Wan2.1-I2V-14B.

## Architecture Differences

| Component | Wan2.1-I2V-14B | Wan2.2-TI2V-5B |
|-----------|-----------------|----------------|
| DiT dim | 5120 | 3072 |
| DiT layers | 32 | 30 |
| DiT heads | 16 | 24 |
| FFN dim | 13824 | 14336 |
| VAE latent channels | 16 | 48 |
| VAE spatial stride | 8× | 16× |
| Model type | i2v | ti2v |

**FFN** = Feed-Forward Network: the two-layer MLP in each transformer block (Linear → GELU → Linear). FFN dim is the intermediate hidden size (e.g. 14336 for 5B).

DreamZero uses a **CausalWanModel** wrapper that extends the base Wan architecture with **action/state registers** for robot policy learning. The same `CausalWanModel` class supports both Wan2.1 and Wan2.2 backbones via configuration—no new class is required. The config switches the architecture parameters (dim, in_dim, out_dim, etc.) and uses `WanVideoVAE38` for the 48-channel Wan2.2 VAE.

**What action/state registers do:** The DiT sees a single sequence `[video_tokens | action_register]` where the action register is encoded action and state features (one chunk per block). All tokens share the same transformer (with causal masking and RoPE). The model learns to predict **video noise** (dynamics) and **action noise** (policy): the action-register slice is decoded by `action_decoder` to produce action noise predictions. So the model is conditioned on current state and (noisy) actions and learns to denoise both video and actions for closed-loop policy learning.

## Causal masking, RoPE, and sequence layout

### Causal masking

In attention, **causal masking** means each position can only attend to **past and current** positions (no future). So token at index `i` can see keys at indices `j ≤ i`. That keeps the model autoregressive: it never uses future video frames or future actions when predicting the current step. In CausalWanModel the masking is **blockwise**: the first frame attends to itself; each later block of video frames can attend to the first frame plus previous (and optionally current) blocks. Action and state tokens have their own causal pattern so each action chunk only sees past video and past actions/state. This matches policy learning where you condition on observed history and predict the next action chunk.

### RoPE (Rotary Position Embeddings)

**RoPE** encodes position by rotating query and key vectors in a complex plane with position-dependent angles. Unlike adding a position vector, RoPE makes attention scores depend on the *relative* position of query and key, which generalizes better to longer sequences. In CausalWanModel:

- **Video tokens** use **3D RoPE**: separate frequency components for frame index (time), height, and width of the patch grid. So each token knows its (t, h, w) in the video.
- **Action and state tokens** use **1D RoPE**: a single position index along the sequence (frame/block index). So the model knows the temporal order of action chunks and state.

Freqs are built in `_create_freqs()` from the patch grid size (F, H, W) and concatenated with separate 1D freqs for the action register.

### Tokens, blocks, and chunks

- **Token**: The smallest unit the transformer sees. After **patch_embedding** (stride 1×2×2 on the latent), one frame yields a 2D grid of tokens; the total per frame is **frame_seqlen** (e.g. 50 for 160×320). So one **token** = one patch (e.g. 1×2×2 in latent space).

- **Block (image block)**: A group of consecutive **frames**, not tokens. **num_frame_per_block** (e.g. 2) frames form one “image block.” So with 33 frames you get multiple blocks. **num_image_blocks** = `(num_frames - 1) // num_frame_per_block`. Blocks are used for blockwise causal attention and to align video with action/state.

- **Chunk**: In policy terms, an **action chunk** is the sequence of actions the policy outputs for one block (e.g. **num_action_per_block** = 24 actions per block). The **action register** in the DiT has one chunk per image block: for each block there are `num_action_per_block` action tokens and `num_state_per_block` state tokens. So the register length is `num_image_blocks * (num_action_per_block + num_state_per_block)`. “Chunk” and “block” are often used together: one video block corresponds to one action chunk (and one state token) in the register.

Summary: **tokens** = patch-level units (50 per frame); **blocks** = groups of frames (e.g. 2 frames per block); **chunks** = per-block action (and state) outputs that are packed into the action register.

## Inference: blocks, chunks, and closed-loop

### How blocks and chunks are used when predicting actions

At **inference**, the model predicts **one action chunk** per call, conditioned on **one block** of video (and current state):

1. **Input**: A short video of the current block — e.g. `num_frame_per_block` frames (e.g. 2) — plus current **state** and (during the denoising loop) **noisy actions** for the chunk being predicted. The first time in a trajectory, the “context” is the first frame (and optionally a warm-up pass with no action to fill the KV cache).

2. **DiT input**: The sequence is `[video_tokens for this block | action_register]`. The action register holds encoded **noisy** actions and **state** for this block only (one chunk). So the DiT sees: “this block of video + this chunk’s noisy actions and state.”

3. **KV cache**: To keep inference causal and efficient, the model uses a **KV cache** over previous blocks. So for the *next* block, the cache already contains keys/values for earlier frames; the DiT only runs on the **new** block’s tokens plus the new action register. `current_start_frame` tells the DiT which block we’re on so RoPE and cache indexing are correct.

4. **Output**: The DiT predicts **video noise** and **action noise**. The action noise is decoded by `action_decoder` into a prediction for the **current chunk**. The scheduler then updates the noisy action toward clean; after `num_inference_steps` denoising steps you get **one clean action chunk** (e.g. 24 actions).

So: **one block of frames** (and state) in → **one action chunk** out. Blocks and chunks are aligned: one image block ↔ one action chunk in the register.

### How DreamZero does closed-loop inference

Closed-loop execution reuses the same block/chunk logic in a loop:

1. **Observe**: Robot has current observation (e.g. image history + state). The policy is called with this observation (e.g. via `lazy_joint_video_action` or `lazy_joint_video_action_causal`).

2. **Predict**: The action head runs the diffusion loop for the **current block**: it encodes the observed frames to latent, runs the DiT (with KV cache and `current_start_frame`) for each denoising step, and returns one denoised **action chunk** (e.g. 24 actions).

3. **Execute**: The robot **executes** that chunk (e.g. 24 steps at 5 Hz → ~4.8 s). No new model call during execution.

4. **Repeat**: After execution, new observation is available. The policy is called again with the new video (e.g. last N frames). If the task/language is unchanged, `current_start_frame` is incremented by `num_frame_per_block` and the KV cache is reused; the DiT only processes the **new** block and predicts the **next** action chunk. If the task or language changes (or the cache is full), the cache and `current_start_frame` are reset.

So closed-loop = **repeated “one block in → one chunk out”** with KV cache across steps so the model never re-processes past frames.

### What changes when you swap the backbone to 5B

The **inference algorithm and API stay the same** for 14B vs 5B:

- Same **block/chunk layout**: `num_frame_per_block`, `num_action_per_block`, `num_state_per_block` (and thus one block → one chunk) are defined by config and data; they do not depend on which backbone (14B vs 5B) you use.
- Same **closed-loop flow**: `lazy_joint_video_action`, KV cache, `current_start_frame`, and the denoising loop are in the **action head** and are shared. The policy still calls the same methods (`get_action`, `lazy_joint_video_action`, etc.).
- Same **backbone role**: The backbone only produces conditioning (e.g. text embeddings). The action head owns the DiT, VAE, and action/state encoders. So “swapping to 5B” means swapping the **action head config** (and checkpoints) to the Wan22 5B DiT + VAE38 + 160×320; the high-level inference path (backbone → action_head → one chunk) is unchanged.

What **does** change with 5B:

- **DiT size and layout**: 5B uses a smaller DiT (dim 3072, 30 layers, 24 heads), **frame_seqlen = 50** (for 160×320), and **no** first-frame latent concat (`concat_first_frame_latent=False`). First frame is conditioned via **CLIP** in the context, not as extra channel in the latent.
- **VAE and resolution**: 5B uses **WanVideoVAE38** (48 channels, 16× spatial) and **160×320** video. So latent is 10×20; tokens per frame = 50.
- **Conditioning**: 5B uses CLIP image embedding for the first frame in the context; 14B can concatenate the first-frame latent to the DiT input. The action head handles this inside the same `_forward_inference` / `_forward_blocks`; no change to the external inference API.

So: **blocks and chunks** are used the same way for predicting actions at inference; **closed-loop** is the same loop of “observe → predict one chunk → execute → repeat” with KV cache; **swapping to 5B** keeps that flow and only changes the internal model (DiT/VAE) and resolution/conditioning.

## Prerequisites

1. **Wan2.2-TI2V-5B** weights:
   ```bash
   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
   ```
   Or clone from [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2) and follow their download instructions.

2. **Image encoder (CLIP)**: Wan2.2-TI2V-5B does not include the CLIP image encoder. Use the one from Wan2.1:
   ```bash
   huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
   ```
   Only `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` is needed.

3. **DROID dataset** in LeRobot format:
   ```bash
   huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
   ```

## Quick Start

```bash
# Set paths (optional - defaults shown)
export WAN22_CKPT_DIR=./checkpoints/Wan2.2-TI2V-5B
export IMAGE_ENCODER_DIR=./checkpoints/Wan2.1-I2V-14B-480P  # for CLIP only
export DROID_DATA_ROOT=./data/droid_lerobot

# Run training
bash scripts/train/droid_training_wan22.sh
```

## Configuration Details

The Wan2.2 config (`wan_flow_matching_action_tf_wan22.yaml`) overrides:

- **model/dreamzero/action_head**: `wan_flow_matching_action_tf_wan22`
- **diffusion_model_cfg**: Wan2.2 architecture (dim=3072, in_dim=48, out_dim=48, etc.)
- **vae_cfg**: `WanVideoVAE38` (48-channel Wan2.2 VAE)
- **frame_seqlen**: 50 (patch output per frame)
- **target_video_height / target_video_width**: 160 and 320 so latent spatial size is **even** (10×20 after VAE38 16×), avoiding a dynamics-loss crop. Previously 176×320 gave latent 11×20 (odd height); we use **160×320** (H×W) so both latent dimensions are even after the DiT’s stride-(1,2,2) patch embedding.

For other resolutions, `frame_seqlen` must match patch output per frame; use H and W divisible by 32 for even latent:
- 160×320 (H×W): latent 10×20 → 50
- 176×320: latent 11×20 → 50 (odd H; loss uses crop)
- 640×352: 220

## Using with Custom Training Scripts

To use Wan2.2 in your own training script, add:

```bash
model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
dit_version=$WAN22_CKPT_DIR \
text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth
```
(Do not pass `frame_seqlen`; the Wan22 config uses 50.)

## File Layout

```
dreamzero/
├── groot/vla/configs/model/dreamzero/action_head/
│   ├── wan_flow_matching_action_tf.yaml      # Wan2.1 (default)
│   └── wan_flow_matching_action_tf_wan22.yaml  # Wan2.2-TI2V-5B
├── scripts/train/
│   ├── droid_training.sh           # Wan2.1 backbone
│   └── droid_training_wan22.sh     # Wan2.2 backbone
└── docs/
    └── WAN22_BACKBONE.md          # This file
```

The action head (`wan_flow_matching_action_tf.py`) automatically detects Wan2.2 vs Wan2.1 based on `in_dim` (48 vs 16) and `vae.z_dim` (48 vs 16), and loads the correct checkpoint files from the appropriate HuggingFace repos when local paths are not found.
