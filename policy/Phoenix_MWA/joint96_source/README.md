# Phoenix MWA Joint96 inference dependencies

This directory is part of the XPolicyLab **policy** submission. It contains the WAM model and sampling code, frozen DINO/STVAE encoders, image/state processors, `inference.json`, embodiment registry, ARX X5/Piper/Piper X robot profiles and normalization, checkpoint provenance and UMT5 tokenizer. Model weights are downloaded individually by `../download_joint96_weights.sh`.

The dependency subset excludes trainers, dataset loaders, preprocessing/cache generation pipelines, optimizer state and training launchers. Shared upstream model classes retain their definitions to preserve checkpoint compatibility. `normalization/manifest.json` and `cache_metadata/` are small provenance records used to verify the checkpoint-to-normalization chain; they contain no training data or feature-cache shards.

`SOURCE_PROVENANCE.json` records retained source hashes and the few integration edits. The WAM factory has an explicit complete-checkpoint construction option; the adapter immediately restores every policy tensor strictly. The StarVLA package exports only the two frozen encoder classes needed here. Original license and copyright notices are retained.

`encoder_source/UPSTREAM_SOURCE.json` is a hash index of the original FastWAM Python tree. The loader checks every retained FastWAM file against that index and reconstructs the tree identity bound into the checkpoint. Omitted training modules are represented only by hashes, never imported or downloaded. DINO's pinned source revision is also checked. `source_manifest.json` verifies every shipped file before model loading.

The source was reduced using the actual GPU inference import inventory. Validation compares the same checkpoint, input and seed before and after reduction. This package exposes no training entry point and does not claim training reproduction.
