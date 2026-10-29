# Services and deploys

A **service** is a model process that skills or model roles call: SAM 3 for
segmentation, Molmo2 for pointing, a VLM, a VLA, OpenWAM, the kinematics. A **deploy**
says where the services of one machine run. Placement never changes what a run
computes, only where and when; the agent decides which services a run uses.

## Manifests

One file per service, `services/<name>.toml`:

```toml
--8<-- "services/sam3.toml"
```

A learned policy needs no server code of ours: protocol `xpolicylab` names the
XPolicyLab policy folder, checkpoint and overrides, and XPolicyLab's own
`setup_policy_server.py` serves it. Any of XPolicyLab's policies becomes a learned skill
with one manifest ([Add a VLA](../tutorials/add-a-vla.md)). The services and their
measured VRAM are in the [services reference](../reference/services/index.md); the
manifest keys in the [configuration reference](../reference/configuration.md#service-manifests).

## Deploys

`configs/deploy/<machine>.toml`: the GPUs the project may use, where weights live, the
simulator workers, and how each service runs. A service the deploy does not list is
off, and the skills that need it drop out of runs on that machine. A deploy file
describes one machine, so it is not part of the project: git ignores every file in
`configs/deploy/` but the `example-*.toml` ones below, and a file elsewhere works as
`--deploy PATH` or `$HARNESS_DEPLOY`.

| Mode | Meaning |
|---|---|
| `resident` | started by `harness up`, kept running |
| `on_demand` | started by the broker when a skill first calls it; stopped after `idle_seconds` or when a peer of its swap group is needed |
| `external` | someone else runs it; only a `url` |
| `off` | not here |

`gpu = "auto"` places a service on the GPU with the most room for its `vram_gb`, largest
first; the members of a swap group go to one GPU and need the room of the largest.

=== "Four GPUs"

    ```toml
    --8<-- "configs/deploy/example-4gpu.toml"
    ```

=== "Two GPUs"

    ```toml
    --8<-- "configs/deploy/example-2gpu.toml"
    ```

=== "One GPU"

    ```toml
    --8<-- "configs/deploy/example-1gpu.toml"
    ```

=== "No GPU for models"

    ```toml
    --8<-- "configs/deploy/example-api.toml"
    ```

## On demand and swap groups

On a machine without a GPU per model, models share GPUs. An on-demand service starts
when a client first asks for it: every client (the policy and world-model clients, the
segmenter, chat roles backed by a service) asks the broker before each call, and the
broker starts the service and waits until it answers. Services of one `swap` group
share a GPU, one loaded at a time; the broker waits while a peer is in use, then swaps.

The simulator does not advance while a model loads (a skill's call simply takes
longer), so swapping costs wall time, not episode steps: the run computes the same
thing as with every model resident, except that a policy with memory (`scoped = true`,
such as Simate) that is swapped out starts its memory afresh.

## Commands

```bash
harness services                       # the manifests
harness doctor --deploy mine           # files, GPUs and free memory, ports, interpreters
harness fetch sam3 simate --deploy mine   # weights into <weights>/<dir> (ModelScope, hf-mirror; sha256-checked)
harness up --deploy mine               # start the resident services (and the broker)
harness status --deploy mine
harness down --deploy mine             # stop what harness up started
```

`harness up` leaves a service alone when its port already answers (started by hand or
by another checkout); `harness down` stops only what it started, by signal escalation
(SIGINT or SIGTERM, then SIGKILL). Logs and pid files go to the deploy's `state`
directory. [Manage services](../how-to/services.md) has the details.

## A deploy written for you

An exported XPolicyLab policy writes its own deploy at start, for the GPUs its policy
server sees: every service on ports of its own (several instances may share a
machine), resident when they fit the free memory, else policies and world models on
demand in one swap group, else every model but the kinematics. The agent's models
(kinematics, SAM 3, Simate: 23.2 GB) fit one 24 GB GPU resident ([Evaluation through
XPolicyLab](xpolicylab.md)).
