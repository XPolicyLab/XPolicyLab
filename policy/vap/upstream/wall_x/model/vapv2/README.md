# VAPv2 unified joint self-attention

Opt-in `model_type: vapv2`, architecture `vapv2.qwen_joint_ivava.v2`.
The legacy `model_type: vap` / T5 model is unchanged.

## Model contract

- Frozen Qwen3.5 (online or versioned raw cache) -> trainable projector -> Text.
- Video width 3072, Action width 1024; independent Video/Action flow timesteps.
- One joint self-attention per block, not condition-joint followed by legacy
  self-attention and text cross-attention. The old cross-attention, text bridge
  and GlobalTimeAnchor are removed from this model's registered parameters.
- Logical sequence: `[Text; Memory; I; A0; V0; A1; V1]`. Physical V/A packing
  is delegated to the original layout. Robust-A0 clean/noisy/robust routing is
  preserved, not approximated by a simple triangular mask.
- Text/valid Memory/I form a bidirectional condition prefix. All generated V/A
  branches see valid Text; prefix queries cannot read ANY generated labels.
- Text has separate Q/K/V/O copied from Wan's self-attention at initialization.
  Text shares the Video-width FFN and uses Video AdaLN at fixed clean time 0.
  Video and Action retain their independent diffusion-time modulation.
- Updated text hidden states are explicit block outputs and next-block inputs,
  including gradient checkpointing. They are not a mutable side-channel.
- Qwen and VAE are frozen; the projector and joint Text/Video/Action backbone
  are trainable. Text predictions do not introduce a new training loss.

## 4D RoPE

Axes stay `(chunk_start, local_time, height, width)` — not camera coordinates.
Camera identity remains a separate embedding. No existing Memory/I/A/V position
is shifted or recalculated. Memory keeps its real negative relative age.

For N valid Qwen tokens, text positions are:
`(-N,0,0,0), ..., (-1,0,0,0)`. Padding positions are zero and masked.
N is the actual valid length, independent of padded capacity. Text Q/K use
exactly the existing 4D RoPE implementation and frequency allocation.
This is a symbolic ordered text lane, not a physical history timestamp.
Its coordinates may overlap memory age; the text projector/type embedding and
separate projections identify the modality. No claim of disjoint coordinates
or empirical superiority is made.

## Memory / Flex / GC

Missing history and padding keys are invisible. Invalid query rows retain only
a self-edge to avoid an empty softmax; their hidden outputs are zeroed each
block. Structural Flex BlockMasks cache only the base layout, text capacity and
device; per-sample validity is an immutable forward-local score_mod snapshot.
A single layout is reused across all blocks and through GC backward.
Memory can read I/Text in VAPv2's condition prefix, unlike legacy read-only
Memory. The physical memory age encoding itself is unchanged.

## Initialization / compatibility

Fresh runs initialize from the configured Wan2.2 base. No pretrain VAP checkpoint
is required. Hybrid-v1 and old VAP checkpoints are NOT exact-resume compatible:
architecture/text-position contracts fail explicitly rather than dropping keys.
Same-version checkpoints may resume subject to existing data/optimizer checks.
The default VAPv2 wrapper run name includes `jointv2` to avoid old output paths.

The legacy action prefix-KV capture/decode API is explicitly unsupported.
Independent full-forward RoboDojo inference is available through
`scripts.vapv2.infer_robodojo` and `VAPv2ServingRuntime`; see
`scripts/vapv2/INFERENCE.md`. It carries explicit request-local Qwen validity
through A0/A1 sampling and does not modify legacy VAP/XPolicyLab dispatch.
Cached decode, robot-loop integration, XM/SOAR and legacy open-loop validation
still require separate implementation/validation.

## Validation

CPU tests cover one-attention topology, multi-layer text propagation, padding
and history perturbation invariance, generated-label isolation, original
robust-A0 submatrix equality, unchanged Memory/A/V RoPE inputs, GC output/gradient
equivalence, unique parameter ownership, real packed training/loss integration,
and DMuon hook-boundary coverage. CUDA Flex-vs-SDPA forward/backward is included
as an optional GPU test in the training source distribution. The eval-only
submission retains the serving structure/inference regressions; it does not ship
training launchers, private validation fixtures or host deployment records.
