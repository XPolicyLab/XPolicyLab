# Use a self-hosted model

Any OpenAI-compatible server can take a model role (`backend = "openai_chat"`), and a
role can name a service of the deploy, which then supplies the URL and the model id.
`services/servers/serve_vlm.py` serves a local vision-language model with transformers
alone (one request at a time; meant for development without API spend).

## The services

```toml
--8<-- "services/qwen3-vl-32b.toml"
```

| Service | Kind | Model | VRAM |
|---|---|---|---|
| `qwen3-vl-32b` | llm | Qwen3-VL-32B-Instruct, 4 bit | 21 GB |
| `qwen3-vl-8b` | vlm | Qwen3-VL-8B-Instruct, bf16 | 18 GB |
| `molmo2` | pointer | Molmo2-8B, 8 bit, vision backbone in bf16 | 12.7 GB |

Place one in your deploy and start it:

```toml
[services.qwen3-vl-32b]
gpu = 4
```

```bash
harness fetch qwen3-vl-32b --deploy mine
harness up qwen3-vl-32b --deploy mine
```

## The model file

```toml
--8<-- "configs/models/qwen3-vl-32b.toml"
```

```bash
harness run configs/experiments/dev.toml --model qwen3-vl-32b --output runs/local
```

A role's own keys go on top of the service's client config (`temperature`,
`timeout_seconds`, `max_output_tokens`, `extra_body`). Self-hosted services report no
cost, so `cost_usd` counts only hosted calls.

## Small context windows

The 32B model in 4 bit runs out of memory above about 7k prompt tokens on a 24 GB GPU.
Its model file keeps the history short: `keep_images = 1`, `max_history_turns = 3`,
and no task descriptions. When a request still runs out of GPU memory, the driver drops
one more turn and retries. `serve_vlm.py --max-image-side N` downscales images on the
server.

## Any other OpenAI-compatible server

```toml
[main]
backend = "openai_chat"
base_url = "http://10.0.0.9:8000/v1"    # vLLM, SGLang, a local proxy, ...
model = "Qwen/Qwen3-VL-235B-A22B-Instruct"
api_key_env = "MY_SERVER_KEY"           # optional; "EMPTY" without one
```
