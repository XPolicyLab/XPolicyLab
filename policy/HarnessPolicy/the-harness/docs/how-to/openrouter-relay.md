# Using an OpenRouter key from another machine

Use this when the OpenRouter API key lives on a different machine (the **key machine**) from the
one that runs the simulator and harness (the **run machine**), and the key should stay where it is.

`scripts/openrouter_relay.py` runs on the key machine. The run machine sends its OpenRouter
requests there with a shared relay token instead of the key. The relay checks the token, replaces
it with its own `OPENROUTER_API_KEY`, forwards the request unchanged to
`https://openrouter.ai/api/v1/...`, and streams the reply back. Model, prompts, sampling settings
and the reply are exactly those of a direct OpenRouter call; the calls are billed to the key
machine's OpenRouter account.

```
run machine                                key machine
harness ──HTTP──> 127.0.0.1:8850 ══SSH tunnel══> relay :8850 ──HTTPS──> openrouter.ai
          (Bearer <relay token>)                  (Bearer <OpenRouter key>)
```

## 1. Key machine: start the relay

The relay needs Python 3.11+ and nothing else, so a copy of `scripts/openrouter_relay.py` is
enough. A full checkout is not needed.

```bash
# Once: make a relay token and keep it for both machines.
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'

export OPENROUTER_API_KEY=<your OpenRouter key>
export OPENROUTER_RELAY_TOKEN=<relay token>
python3 scripts/openrouter_relay.py --port 8850
```

It prints one line per request (time, client, path, model, status, seconds) and never logs prompts
or keys. `--timeout` (default 900 s) bounds a single OpenRouter call.

## 2. Run machine: open an SSH tunnel

The relay speaks plain HTTP and listens on 127.0.0.1 only, so reach it through SSH:

```bash
ssh -N -L 8850:127.0.0.1:8850 <user>@<key machine>
```

Keep it running in a separate terminal (or add `-f` to background it). Do not start the relay with
`--host 0.0.0.0` on a network you do not control: the token and prompts would then travel
unencrypted.

## 3. Run machine: point the harness at the relay

```bash
export OPENROUTER_RELAY_TOKEN=<relay token>
export OPENROUTER_RELAY_URL=http://127.0.0.1:8850/api/v1
```

With `OPENROUTER_RELAY_URL` set, the configuration loader sends **every OpenRouter model
role** through the relay: the main model, the VLM and the verifier. Configuration files
stay unchanged, and `OPENROUTER_API_KEY` is not needed on the
run machine. Roles backed by a service (`service = "molmo2"`), self-hosted `openai_chat`
models and other providers are not touched. Unset the variable to call OpenRouter
directly again.

```bash
harness run configs/experiments/dev.toml --output runs/relay
```

The resolved configuration, and so every episode's manifest, records `base_url` as the
relay URL and `api_key_env = "OPENROUTER_RELAY_TOKEN"`, so a run shows that it went
through the relay. The relay changes only the transport: a run through it is
comparable with a direct run of the same configuration, task, seed and budget, and its
identity is the same.

`OPENROUTER_BASE_URL` instead points every OpenRouter role at another
OpenRouter-compatible URL (a proxy) with the usual `OPENROUTER_API_KEY`. Setting both
variables is an error.

## Without the environment variable

To fix the relay in one agents file only, set it on each OpenRouter role:

```toml
[main]
backend = "openrouter"
base_url = "http://127.0.0.1:8850/api/v1"
api_key_env = "OPENROUTER_RELAY_TOKEN"

[vlm]
backend = "openrouter"
base_url = "http://127.0.0.1:8850/api/v1"
api_key_env = "OPENROUTER_RELAY_TOKEN"
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Set OPENROUTER_RELAY_TOKEN before selecting the openrouter backend` | The token is not exported on the run machine. |
| HTTP 401 `wrong or missing relay token` (with `"relay": true`) | The tokens on the two machines differ. |
| HTTP 502 `OpenRouter unreachable` | The key machine cannot reach openrouter.ai. |
| Connection refused on 127.0.0.1:8850 | The SSH tunnel or the relay is not running. |
| Any other HTTP error | It came from OpenRouter itself and is passed through unchanged. |

The relay does not retry. Retries, timeouts and streaming are still set by the role (`retries`,
`retry_seconds`, `timeout_seconds`, `stream`), as for a direct call.
