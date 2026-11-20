# Retained inference code sources

Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

## Robbyant

The public upstream reference is
[Robbyant/Lingbot-VA at `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`](https://github.com/Robbyant/lingbot-va/tree/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb),
distributed under the [Apache License 2.0](SOURCE_LICENSE.txt).
The license text is copied unchanged from that revision.

### Verified scope

- `models/rotary.py`: the complete `WanRotaryPosEmbed` class, including
  `_precompute_freqs_base`, matches the Python syntax tree in upstream
  [`wan_va/modules/model.py`](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/modules/model.py).
  It is packaged in a different module; its numerical class implementation is
  unchanged. This notice does not attribute the other functions in that module
  solely to Robbyant.
- `train/utils.py` retains the supplied Robbyant copyright header. Its
  `get_mesh_id` and `sample_timestep_id` have public counterparts in
  [`wan_va/utils/utils.py`](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/utils/utils.py).
  The supplied integration omits the `t` argument and fourth coordinate from
  `get_mesh_id`, and adds generator/device arguments to `sample_timestep_id`.
  Only these inference utilities are retained; this is not a release of the
  upstream training pipeline.

## Alibaba Wan

The public reference is
[Wan-Video/Wan2.2 at `1ea34ff48f87168174e12956e200b1d908b1c5ff`](https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff).
Its [Apache-2.0 license](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/LICENSE.txt)
is byte-identical to the retained [license text](SOURCE_LICENSE.txt).
The original Alibaba copyright headers remain in all three files.

- `models/t5/tokenizers.py` is byte-identical to upstream
  [`wan/modules/tokenizers.py`](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/tokenizers.py).
- `models/t5/t5.py` retains all 15 upstream top-level definitions from
  [`wan/modules/t5.py`](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/t5.py).
  Fourteen have identical syntax trees. The supplied `T5EncoderModel` releases
  the temporary checkpoint state dictionary after loading, attempts host-memory
  cleanup, and returns fixed-length padded context instead of trimming each
  sequence. These adaptations are preserved, not replaced by upstream code.
- `models/vae/vae2_2.py` retains all 18 upstream top-level definitions from
  [`wan/modules/vae2_2.py`](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/vae2_2.py).
  Fifteen have identical syntax trees. The supplied adaptations add streaming
  and batched encoding/cache methods, release temporary checkpoint state after
  loading, and use `torch.amp.autocast('cuda', ...)` in the public wrapper.

These comparisons establish the named third-party relationships and explicit
local adaptations. The authors confirm that the project-owned inference code
and adaptations are original work without a separate public upstream, as recorded
in the [project declaration](../../../../docs/third-party-provenance.md).
That declaration does not replace the third-party attributions above. Model
weights, datasets and external visual models retain their separate terms;
this source license does not cover them.
