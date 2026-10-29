# Third-party notices

The G07_preview adapter is licensed under the Apache License 2.0 (`LICENSE`). The files below are derived from
Apache-2.0 projects and modified by Galaxea AI for inference-only use; each carries its original copyright notice and a
modification notice.

| File | Derived from | Original copyright |
| --- | --- | --- |
| `src/galaxea_va/models/cosmos3/vision_encoder.py` | [transformers](https://github.com/huggingface/transformers) `models/qwen3_vl/modeling_qwen3_vl.py` (`Qwen3VLVisionModel`) and three `vision_utils` helpers | The Qwen Team and The HuggingFace Inc. team |
| `src/galaxea_va/models/cosmos3/nn_blocks.py` | [diffusers](https://github.com/huggingface/diffusers) `models/normalization.py`, `models/embeddings.py`, `models/transformers/transformer_cosmos3.py` | HuggingFace Inc.; The HuggingFace Team; The NVIDIA Team and The HuggingFace Team |
| `src/galaxea_va/models/cosmos3/transformer.py` | diffusers `models/transformers/transformer_cosmos3.py` (module and parameter layout) | The NVIDIA Team and The HuggingFace Team |
| `src/galaxea_va/models/cosmos3/mrope.py` | diffusers `pipelines/cosmos/pipeline_cosmos3_omni.py` (mRoPE position ids) | The NVIDIA Team and The HuggingFace Team |
| `src/galaxea_va/models/wan22/wan_video_vae.py` | [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) `diffsynth/models/wan_video_vae.py`, itself based on the [Wan2.2](https://github.com/Wan-Video/Wan2.2) VAE | DiffSynth-Studio contributors; The Alibaba Wan Team Authors |

Model weights downloaded by `download_weights.sh` carry their own licenses and notices next to the files under
`checkpoints/`.
