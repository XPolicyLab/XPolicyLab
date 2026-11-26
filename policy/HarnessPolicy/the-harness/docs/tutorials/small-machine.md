# Run on a small machine

The agent's models take 21.2 GB of GPU memory resident: Simate 14 GB and SAM 3 7.2
(the main model and the VLM are hosted), plus 6-9 GB per simulator worker. This
tutorial runs it on two GPUs, one GPU, and no GPU for models. The trade is always the
same: the run keeps the configuration and its identity and takes longer (on one GPU a
swap also empties the policy's memory), or it runs a smaller configuration with another
identity.

## Two GPUs: everything resident

```toml
--8<-- "configs/deploy/example-2gpu.toml"
```

```bash
harness up --deploy example-2gpu        # starts SAM 3 and Simate on GPU 1
harness config configs/experiments/dev.toml --deploy example-2gpu
```

## One GPU: swap the two models

```toml
--8<-- "configs/deploy/example-1gpu.toml"
```

The simulator leaves room for one model at a time, so SAM 3 and Simate are on demand in
one swap group: the broker loads the one a skill calls and swaps when the other is
needed (`locate` and the scene report after every program call SAM 3; a handed-over task
calls Simate). The simulator waits while a model loads, so this costs wall time, not
episode steps: the configuration and its identity are those of the two-GPU machine.

One thing is not the same. Simate keeps a memory of past camera frames, and a policy
with memory that is swapped out starts it afresh. When a program calls SAM 3 between two
runs of the policy in one episode (the scene report after a stalled handover, a `locate`
in the calls after it), the next run reloads Simate without the frames before. Tasks that
need the memory (an object seen, covered or swapped earlier) can score differently, so
compare one-GPU runs with one-GPU runs.

A swap group needs the room of its largest member (14 GB here) next to everything
resident on that GPU. `harness doctor --deploy <file>` checks the plan against the free
memory.

## No GPU for models

```toml
--8<-- "configs/deploy/example-api.toml"
```

Only the simulator runs here; every model role is hosted. Without a segmenter,
segmentation and the tools built on it (`locate`, `pick`, `release`, ...) drop out, and
without a policy server the learned skills and the strategy step:

```text
segmentation  no   no segmenter service
eef           no   no segmenter service
vla           no   no policy service
wam           no   no world_model service
```

That is a much smaller configuration (another identity): useful for checking prompts
and the driver, not for results.

## Molmo2 and OpenWAM

The agent points with its VLM and has no world model. Experiments that want them run
the services (`configs/deploy/example-4gpu.toml` puts both on demand on a GPU of their
own: Molmo2 12.7 GB, OpenWAM 15.3) and add them to the agent, for another identity:

```bash
harness run configs/experiments/dev.toml --deploy mine --output runs/dev-wam \
  --set 'agent.services.world_models=["openwam"]' \
  --set 'agent.roles=["main", "vlm", "pointer", "verifier"]'
```

## Self-hosted models instead of API calls

The `qwen3-vl-32b` model file runs Qwen3-VL-32B in 4 bit on one 24 GB GPU as the main
model and the VLM, with no API spend. [Use a self-hosted model](../how-to/self-hosted-models.md).
