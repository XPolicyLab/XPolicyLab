# Wan2.2 distributed inference contract

Wan2.2 WAM distribution is explicit and fail closed. A `torchrun` environment
does not enable it. Multi-rank construction requires all of:

- `Wan22DistributedConfig(backend="torch", image_parallel_size=...)`;
- an initialized, matching process group through `Wan22Collective`;
- a `Wan22ImageParallelProtocol` implementation;
- every rank entering plugin, kernel, denoising-step, and core-block collectives
  in the same order.

`Wan22DistributedContext` owns the process-group/device-mesh view, validates
world size and rank membership, identifies the coordinator, and verifies
collective ordering. Random video/action noise is created by the coordinator
and broadcast. KV and cross-attention caches remain rank-local; committed
`WAMInferenceState` records its owner rank and mesh world size and cannot be
reused by another rank or a resized mesh.

The plugin attaches the non-owning context to the unified `CausalWanModel`; the core
calls the image-parallel protocol around each inference transformer block.
Single-rank behavior remains the default and does not execute collectives.
Multi-rank training is rejected.

## Current limitations

- This repository defines and integrates the image-parallel protocol, but does
  not ship a production CUDA token/head-sharding implementation.
- `ReplicatedImageParallel` is only a CPU/gloo or mocked-collective correctness
  backend. It duplicates model compute and memory and is not a performance
  backend.
- Real checkpoint loading, GPU execution, checkpoint conversion, training, and
  evaluation parity are outside this implementation.
- A torch process group without an image-parallel protocol is still rejected.
- The native websocket server remains single-rank unless its caller supplies a
  fully constructed matching `Wan22DistributedContext`; CLI launch does not
  infer or auto-create one from environment variables.
