# RoboDojo policy server

This server adds isolated client sessions, fixed shrinking batches and optional
WebSocket replay to the existing WSP adapter. It supports `arx_x5 / joint`,
interactive flash (H24/C1) and memory (H24/C24) checkpoints.

## Start

Install the policy with `../install.sh` first. From `policy/WorldScape_RSI/`:

```bash
export WSP_PYTHON=/path/to/policy-env/bin/python
bash serve_robodojo.sh flash --checkpoint /path/to/flash-checkpoint --gpus 0
bash serve_robodojo.sh memory --checkpoint /path/to/memory-checkpoint --gpus 0,1
```

Run these separately, or use disjoint GPUs and different ports. The default
address is `0.0.0.0:18880`. `--host`, `--port` and `--run-dir` are configurable.
GPU selection defaults to `CUDA_VISIBLE_DEVICES`, or GPU 0 when it is unset.
Selected GPUs must be idle; the launcher never stops existing processes.
This is a foreground service; Ctrl-C stops its workers.

Both profiles use `../wam_deploy_robodojo.yml`. `--config` selects an alternative
deployment YAML, and `--checkpoint` overrides its `checkpoint_path`.
The profile checks checkpoint geometry rather than changing model parameters:
normalization comes from the transform bundle, history capacity from model
metadata, and preprocessing from provenance. Legacy memory q99 checkpoints
remain supported alongside z-score checkpoints. Gripper thresholds remain in
YAML; the checkpoint controls whether discretization is enabled.
No task-specific prompt enhancement is added.

Each selected GPU holds one weight replica and serves at most one active
session. `--max-sessions` may reduce the number of admitted sessions.
`--envs-per-session` accepts 1–10 and defaults to 3; it limits a client's batch,
not the number of weight replicas. Capacity 10 is a protocol limit, not a GPU
memory guarantee. Validate your checkpoint, history length and hardware before
increasing it.

## Client contract

The transport is the repository's binary WebSocket/MessagePack protocol.
Use `client_server.ws.model_client.WsModelClient` with a unique
`evaluation_id + trial_id` for every episode.

- The default `official / isolated` mode accepts the official single-environment
  loop: reset, observation, prediction, then one observation for every executed
  action. The final observation of a full chunk also starts the next chunk.
- Batch clients can import
  `XPolicyLab.policy.WorldScape_RSI.robodojo_server.batch_client.eval_one_episode_batch`.
  Every observation must carry a stable `env_idx`. A batch may shrink during an
  episode; adding or reintroducing environments requires reset or a new trial.
- End episodes with `TRIAL_END` (`client.call("trial_end", result)`). Disconnecting
  the socket alone leaves the trial eligible for recovery during the grace period.
  Ended trials cannot be reopened; use a new `trial_id`.
- The shared server decodes incoming camera colors to RGB before passing them
  to the adapter. Checkpoint-specific BGR input remains an adapter setting.

The official Agent evaluation client in `../deploy.py` uses reset, observation
updates and action requests. For direct WAM execution, follow the client contract
above. `--client-mode explicit --session-mode legacy` is reserved for clients
that send explicit commit/discard RPCs; it does not isolate concurrent trials
or provide the replay extension.

## Preflight, identity and runtime artifacts

```bash
bash serve_robodojo.sh memory --checkpoint /path/to/memory-checkpoint --gpus 0,1 --check
# Use the same options as the running service:
bash serve_robodojo.sh memory --checkpoint /path/to/memory-checkpoint --gpus 0,1 \
    --verify ws://127.0.0.1:18880
```

`--check` validates imports and checkpoint metadata without loading GPU weights.
It does not prove inference readiness or verify every weight checksum. Model
loading retains the adapter's `validate_checkpoint_artifacts` checks.

Every invocation prints `ONLINE_RUN`. Its unique directory contains
`server.yml`, `preflight.json`, `status.json`, and, after loading,
`effective_model_config.json`. Temporary files and Triton/CUDA caches also live
there. Isolated-session worker Triton caches are separate. The default run parent is
`${XDG_CACHE_HOME:-$HOME/.cache}/worldscape-policy/server`; set `--run-dir` for
another writable location. Model asset caches continue to use the existing
model environment. Capture stdout/stderr for persistent service logs.

## Recovery and validation

See [WebSocket replay](WEBSOCKET_REPLAY.md) for negotiation, request identity,
connection takeover, limits and terminal states. Recovery records are in memory;
restarting the service invalidates them. `POLICY_REPLAY_TOKEN` optionally adds
authentication to the recovery/inspection extension.

The bundled [Harness CPU tests](../harness/tests/) require the dependencies in
`../requirements-test.txt`. From the XPolicyLab parent workspace:

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
export R2SI_TEST_XPL_ROOT="$PWD/XPolicyLab"
python -m pytest -q XPolicyLab/policy/WorldScape_RSI/harness/tests
python -B XPolicyLab/policy/WorldScape_RSI/harness/tests/protocol_smoke.py
```

These check the Agent/WAM interface and official Agent WebSocket transport
using CPU fixtures, including RGB formats and trial ownership. They do not
validate this direct WAM server's isolated-session replay, worker failure
recovery, GPU capacity or robot task success.
