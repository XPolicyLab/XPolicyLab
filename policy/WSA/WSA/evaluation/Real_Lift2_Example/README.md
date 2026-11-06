# Real Lift2 Example

This example shows a Lift2 real-robot setup for WSA. The flow is split
across two machines:

- `serve`: loads the checkpoint and answers websocket requests.
- `run`: reads ROS observations and executes returned action chunks.

## Entry Points

- `01_serve_wsa_base_real_lift2.sh`: start the WSA-Base server.
- `01_serve_wsa_large_real_lift2.sh`: start the WSA-Large server.
- `inference_wsa_large_sync.sh`: WSA-Large server/client launch example.
- `run_real_lift2_inference.sh`: start the robot-side loop.
- `02_inference_lift2.sh`: one-shot launcher for the full stack.
- `test_remote_server.py`: connectivity smoke test.
- `remote_client.py`: lightweight client wrapper for custom loops.

## Runtime

- State and action are 14D.
- Request images are `cam_high`, `cam_left_wrist`, and `cam_right_wrist`.
- The first successful inference waits for manual safety confirmation.
- `DISABLE_3D_TEACHER_FOR_EVAL=true` is the default.
- `STATS_PATH` is optional; if omitted, the server uses `CHECKPOINT_DIR/stats.json`.

## WSA-Base Quick Start

Serve:

```bash
CHECKPOINT_DIR=/path/to/WSA/checkpoints/060000 \
ACTION_MODE=abs \
INFER_HORIZON=50 \
bash evaluation/Real_Lift2_Example/01_serve_wsa_base_real_lift2.sh
```

`CHECKPOINT_DIR` can point to a checkpoint step dir or directly to
`pretrained_model/`. `ACTION_MODE=delta` is also supported when it matches the
checkpoint.

Run:

```bash
WS_URL=ws://127.0.0.1:8000 \
PROMPT="Clear the junk and items off the desktop." \
FRAME_RATE=24 \
IMAGE_HISTORY_INTERVAL=15 \
INFERENCE_MODE=sync \
bash evaluation/Real_Lift2_Example/run_real_lift2_inference.sh
```

Only set `SEND_IMAGE_HEIGHT` and `SEND_IMAGE_WIDTH` if you want a
bandwidth/latency tradeoff.

## WSA-Large Quick Start

Use `inference_wsa_large_sync.sh` as the WSA-Large template. Before running it,
replace its checkpoint, prompt, websocket URL, and text-cache paths with values
for your deployment. For a plain-text bring-up without a cache, start
`01_serve_wsa_large_real_lift2.sh` with
`WSA_LARGE_LOAD_TEXT_ENCODER=true`.

## Common Options

Serve side:

- `CHECKPOINT_DIR`: checkpoint step directory or `pretrained_model/` directory
  loaded by the websocket server.
- `HOST` and `PORT`: server bind address. Defaults to `0.0.0.0:8000`.
- `ACTION_MODE`: action representation expected by the checkpoint, usually
  `abs` for Lift2 checkpoints unless the checkpoint was trained with delta
  actions.
- `INFER_HORIZON`: number of future actions returned per request. The quick
  start uses `50`.
- `STATS_PATH`: optional explicit path to `stats.json`. If omitted, the server
  uses `CHECKPOINT_DIR/stats.json`.
- `STATS_KEY`: stats entry to read when the checkpoint stores multiple robot
  stats.
- `QWEN3_VL_PRETRAINED_PATH`, `QWEN3_VL_PROCESSOR_PATH`, and
  `COSMOS_TOKENIZER_PATH_OR_NAME`: WSA-Base-only overrides; use them only when
  loading local copies of those assets.
- `WSA_LARGE_LOAD_TEXT_ENCODER` and `WSA_LARGE_TEXT_EMBED_CACHE_DIR`:
  WSA-Large text-conditioning mode.

Robot side:

- `WS_URL`: websocket server URL, for example `ws://192.168.1.10:8000`.
- `PROMPT`: language instruction sent with every inference request.
- `FRAME_RATE`: robot control loop rate. The default example uses `24`. That means 24 actions are poped out in a second.
- `IMAGE_HISTORY_INTERVAL`: camera history spacing in control frames. Keep this
  aligned with the checkpoint setup.
- `SEND_IMAGE_HEIGHT` and `SEND_IMAGE_WIDTH`: optional resize before network
  transfer. Leave unset to send native camera resolution.
- `INFERENCE_MODE`: request scheduling mode. Use `sync` for the simplest
  blocking loop; use the defaults unless you are testing latency-aware modes.

## Smoke Test

```bash
python evaluation/Real_Lift2_Example/test_remote_server.py --ws_url ws://127.0.0.1:8000
python evaluation/Real_Lift2_Example/test_remote_server.py --ws_url ws://127.0.0.1:8000 --smoke_infer
```

## Notes

- If you enable the 3D teacher, make sure the serve env can import
  `depth_anything_3`.
