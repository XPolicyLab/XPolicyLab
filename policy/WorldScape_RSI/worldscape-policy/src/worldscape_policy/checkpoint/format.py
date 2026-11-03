"""Native checkpoint artifact format constants."""

from worldscape_policy.checkpoint.transforms import TRANSFORM_BUNDLE_FILENAME

NATIVE_ARTIFACT_SCHEMA_VERSION = "1"
REQUIRED_NATIVE_ARTIFACTS = (
    "checkpoint_manifest.json",
    "provenance.json",
    "model_config.yaml",
    "generation_config.yaml",
    "normalization.json",
    TRANSFORM_BUNDLE_FILENAME,
)
