# Native websocket policy server

The WorldScape-owned server runs a checkpoint through
`build_wan22_policy_from_checkpoint` and `PolicyRuntime`. It is single-rank:
do not launch it with `torchrun`.

Install the transport dependency and launch:

```bash
pip install -e ".[server]"
wsp-serve \
  --checkpoint-dir /path/to/checkpoint \
  --visual-input-range zero_one \
  --device cuda \
  --port 8000
```

`wsp-serve` is the only published server command. It accepts native checkpoint
artifacts only; convert older checkpoints before serving.

`--visual-input-range` is required and must be `uint8`, `zero_one`, or
`minus_one_one`. The declaration applies to `images`, `head_view`,
`goal_images`, and `demo_videos`; requests outside that numeric range fail
before inference.

## Protocol

The server first sends a msgpack metadata frame. After that, it accepts either
JSON text frames or msgpack binary frames and responds using the request's
encoding. Msgpack is recommended for ndarray payloads.

Every request is an object with an `operation` field:

```json
{"operation":"reset","mode":"interactive","seed":0}
```

```json
{
  "operation": "predict",
  "observation": {
    "images": "ndarray [B,T,V,C,H,W]",
    "head_view": "ndarray [B,1,C,H,W]",
    "proprioception": "ndarray, batch first",
    "embodiment_id": "int64 ndarray [B]"
  },
  "prompts": {
    "language_instruction": ["instruction for Interactive mode"],
    "vlm_planning_text": null,
    "negative_language_instruction": null,
    "negative_vlm_text": null,
    "planning_labels_text": null,
    "goal_images": null,
    "demo_videos": null
  }
}
```

A successful prediction returns `result.prediction_id`, `result.action`, and
scalar/tensor metrics. Video is omitted unless the server starts with
`--include-video`. The client must resolve each prediction before requesting
another:

```json
{"operation":"commit","prediction_id":"id returned by predict"}
```

Use `commit` only after the action was accepted/executed; this advances policy
memory. If execution was rejected, use `discard`, which leaves committed
memory unchanged:

```json
{"operation":"discard","prediction_id":"id returned by predict"}
```

For migrated real-robot conditioning, send visual data only on the first
`predict` after `reset`. Text-only uses an empty visual prompt:

```json
{"prompts":{"vlm_planning_text":["Fold the shirt."],"goal_images":null,"demo_videos":null}}
```

Goal conditioning uses one head frame (`[B,1,C,H,W]`):

```json
{"prompts":{"vlm_planning_text":["Build the blocks."],"goal_images":"ndarray [1,1,3,H,W]","demo_videos":null}}
```

Demo50 conditioning is exactly 50 uniformly sampled frames and either one head
view or three views in high/left/right order:

```json
{"prompts":{"vlm_planning_text":["Complete the demonstration."],"goal_images":null,"demo_videos":"ndarray [1,50,1|3,3,H,W]"}}
```

Later predictions in the same session must set both visual fields to null. A
new external upload requires a new `reset`; this clears visual, WAM, and event
state before the replacement prompt is sent.

Responses always contain `ok`, `operation`, and a `record` with status and
end-to-end handler latency in milliseconds. Failures additionally contain a
stable `error.type` and `error.message`; protocol errors do not expose a
traceback or terminate the websocket.
