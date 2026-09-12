# WorldScape Policy migration architecture

This repository is being migrated without changing the DreamZero numerical core in
the first stage.

## Public boundary

`worldscape_policy` now defines the new stable boundary:

```text
ObservationBatch + PromptBatch
          |
    ConditionRouter
      /         \
 AutoConditioner InteractiveConditioner
          |
 VisualPrefillManager
          |
       WAMPlugin
          |
  WorldActionOutput
```

The mode router is the only component allowed to select between Auto and
Interactive language conditioning. Goal images and demonstration videos are
encoded as persistent visual prefill; they are never appended to language
cross-attention tokens.

## Current migration stage

Implemented in this stage:

- explicit policy, prompt, condition, event-memory, visual-memory, and output types;
- mutually exclusive condition routing;
- persistent and recent visual-memory lifecycle;
- a bounded event-memory FIFO with explicit episode reset semantics;
- condition-agnostic WAM plugin protocol;
- native Auto/Interactive conditioner orchestration with a shared-pass VLM
  feature contract and explicit T5 text-encoder contract;
- shared, training-only semantic-target selection in
  `conditioning/semantic_forcing.py`;
- explicit event-boundary selection, memory retrieval, and memory gating
  adapters with legacy-exact operation order and checkpoint keys;
- native batch collation in `data/collate.py`, with compatibility imports from
  `data/transform.py`;
- a top-level policy orchestrator for training and sampling;
- a native single-rank Wan2.2 numerical kernel with explicit positive/negative
  CFG conditions and transactional causal caches;
- a local HDF5 replay path using `PolicyRuntime`.

Native checkpoint bundles use format v2 with checksum-validated artifacts.
Runtime construction, training, evaluation, and serving load native modules
exclusively from `worldscape_policy.*` component targets.

## Invariants

1. Auto requires `vlm_planning_text`; Interactive requires
   `language_instruction`.
2. Interactive never receives or returns event memory.
3. `semantic_target` may only exist during training.
4. Semantic forcing is evaluated only on Auto-routed samples. Interactive
   samples use T5 conditioning for their WAM video/action flow losses and never
   trigger an auxiliary Qwen forward.
5. Goal image and demo video are mutually exclusive per sample. A training
   batch may contain both tensors when explicit sample masks are provided.
6. Persistent visual prompt, recent observations, positive/negative WAM KV
   caches, and positive/negative cross-attention caches have separate fields
   and reset semantics.
7. Legacy checkpoint mode is read from the model and cannot be changed by an
   inference override.

## Native module ownership

Checkpoint keys, freezing rules, and builders must target the same registered
module tree:

```text
WorldScapePolicy
├── condition_router
│   ├── auto
│   │   ├── vlm
│   │   │   ├── vlm
│   │   │   └── qformer
│   │   ├── token_pooler (non-Qwen pooled-feature adapters only)
│   │   ├── projector
│   │   └── event_memory
│   └── interactive
│       └── t5
├── visual_memory
│   └── codec
│       └── vae
└── wam
    ├── image_encoder
    └── core
```

`VisualPrefillManager` is an `nn.Module` because it owns the registered visual
codec as well as the non-parameter episode-memory lifecycle. Runtime
`EventMemoryState` remains separate from the learned Auto conditioner
`event_memory` module.

Wan2.2 additionally requires a raw reference frame to build its window-stable
I2V condition (`clip_features` plus masked VAE latents). This condition is not
interchangeable with recent observation latents. It is stored with the WAM
causal state and advances only when `PolicyRuntime.commit()` accepts a sampled
action. The plugin owns `image_encoder` and `core`; the shared VAE remains
registered only under `visual_memory.codec.vae`.

The migrated Qwen adapter defaults to the final VLM layer and does not construct
QFormer in that mode. One CoT-wrapped multimodal prefill produces the perception sequence
`[B,L,D_vlm]`; cached autoregressive decoding then contributes planning-token
states `[B,P,D_vlm]` from that same final-layer feature space. The two sequences
are concatenated along `L+P` and one shared projector maps `D_vlm` to the WAM
cross-attention width `D_condition`.

With `VLM_TOKEN_MODE=qformer`, QFormer is constructed under
`condition_router.auto.vlm.qformer`; selected prefill hidden layers enter it as
key/value features and learned query tokens produce a compressed
perception sequence `[B,Q,D_vlm]`. Autoregressive planning tokens do not pass
through the QFormer: they remain final-layer Qwen states `[B,P,D_vlm]`.
Consequently `qformer_output_dim` must equal the Qwen hidden/token dimension;
the WAM projection still happens once, after perception and planning are joined.
Registering the QFormer again as `auto.token_pooler` would duplicate checkpoint
ownership and apply pooling twice.

The builder also requires an explicit `visual_input_range` (`uint8`,
`zero_one`, or `minus_one_one`). Range inference is intentionally forbidden:
an all-positive tensor cannot reveal whether it has already been normalized.

The checkpoint migration changes the learned memory's owner but deliberately
preserves its internal parameter names. For example,
`action_head.latent_cot_memory.gist_query_proj.*` becomes
`condition_router.auto.event_memory.gist_query_proj.*`. This keeps old weights
loadable while allowing the implementation class itself to be renamed.

`EventMemoryFusion` exposes `EventBoundarySelector`, `MemoryRetriever`, and
`MemoryGate` as stateless composition stages. They call the existing learned
projections on the owning memory module, so no new state-dict keys are added and
the returned diagnostic keys remain unchanged.

The visual-memory module is an explicit constructor dependency of
`WorldScapePolicy`. A WAM plugin cannot be reused implicitly as the visual
codec; otherwise the same module would be registered under both `wam.*` and
`visual_memory.codec.*`, making native checkpoint names ambiguous.

## WAM plugin registry

`worldscape_policy.wam.registry` is the only native WAM construction boundary.
Plugins are registered explicitly by a config type and a stable name, version,
and capability set. The registry never resolves arbitrary import strings or
loads entry points. Callers can require capabilities before construction, so an
unsupported backend fails before checkpoint or device work begins.

The built-in registry currently contains:

- `wan22` version `2.2`, with native training, sampling, causal-cache, and image
  conditioning capabilities;
- `wan21` version `2.1`, as a protocol adapter shell only. Its sampling and
  training methods raise clear `NotImplementedError` exceptions until a native
  implementation is supplied.

Wan plugin packages contain model/protocol code only and do not choose an
implementation from operating-system branches. Platform and deployment choices
belong outside the model package.

The Wan2.2 factory receives the existing core, image encoder, numerical kernel,
and a non-owning visual-codec provider. Registry construction therefore leaves
the checkpoint tree unchanged: `wam` owns only `core` and `image_encoder`, while
the VAE remains owned once by `visual_memory.codec`.

## Unified native pretraining conditions

Native pretraining uses one canonical Wan2.2 WAM parameter tree for
T2VA, goal-image-to-VA, and video-to-VA. These are `ConditionMode` values, not
different action heads or plugins. T2VA supplies no persistent visual prefill;
goal and video prompts are encoded by the one registered visual codec and
passed to the same WAM as `persistent_prefill`.

`ConditionMode` is orthogonal to Auto/Interactive language routing. Mixed
batches are partitioned by interaction mode, visual condition, and demo length
before forward. Every partition calls the same policy and WAM objects, then the
trainer restores original sample order before applying the objective. This
preserves the current scalar permanent-context-length contract without
duplicating parameters. Legacy checkpoints are conversion inputs only; no
legacy Python action head is a native training or runtime extension point.

## Inference state transaction

`PolicyRuntime` owns episode state and uses a two-phase transition:

```text
reset(mode)
    |
predict(observation, prompt)
    |
pending action + candidate next memory
    |                         |
commit after execution       discard on failure
    |                         |
advance event/visual state   retain previous state
```

This prevents an unexecuted action from advancing long-term event history or
short-term visual prefill. Simulator and robot backends should depend on this
runtime rather than storing model caches themselves.

Auto inference keeps current-window event tokens pending and reuses one cached
condition. At a local-attention rollover, pending tokens are promoted into the
bounded long-term FIFO and the condition is refreshed. A single-frame reset or
prompt change clears the old event window instead. Replacing a persistent
goal/demo prompt also increments its visual-memory version and invalidates WAM
caches before the new prompt can be used.

## Deliberate differences from the initial directory sketch

The repository follows the target subsystem boundaries but keeps several
intentional improvements over the initial file tree:

- Checkpoint construction and loading remain centralized in
  `checkpoint/loader.py` and `native_builder.py`. A Wan22-local
  `pretrained_loader.py` would duplicate the policy-level loading contract.
- `data/adapters/hdf5.py` is named for its actual native `EventSample` output;
  only `legacy_context.py` understands legacy context fields.
- Checkpoint conversion and parity tests stay under unit/integration according
  to test scope instead of restoring a runtime-oriented `tests/compatibility`
  package.
- Generic websocket serving and robot evaluation remain documented separately
  in `server.md` and `evaluation.md`.
- `evals/common` keeps explicit protocol, schema, environment, evaluator, and
  backend modules instead of collapsing these concerns into generic
  `env.py`, `metrics.py`, and `rollout.py` files.
