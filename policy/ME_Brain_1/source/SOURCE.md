# Source provenance

Upstream: https://github.com/MachEmbodied/Focus-VLWA

Base commit: `97aa6162f9d6c6c72ab9e0f6850ce87fbc7733b7`.

This ordinary directory contains model, inference, dataset and post-training source,
upstream tests, package metadata, and license notices. It contains no nested Git
repository, checkpoint, virtual environment, or demo video. The local source
checkout's README changes are retained and adjusted to the bundled adapter path.

Integration changes:

- Both post-training entry points share stage, resume and precision arguments.
- The training seed is exposed as `--seed` and forwarded to `PostTrainingConfig`.
- Training dependencies load after argument parsing so help works without LeRobot.
- Trailing whitespace in the Gemma patch and license text is normalized without changing their content.

Apache-2.0 and Gemma notices are preserved in `LICENSE`, `LICENSE_GEMMA.txt` and `NOTICE`.
