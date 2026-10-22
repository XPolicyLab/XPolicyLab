"""Private persistence identifiers for the artifact cache."""

from __future__ import annotations


# Persistence details stay inside the cache package.  They are intentionally
# not exported from ``wam.cache`` or exposed to training code.
FORMAT_VERSION = 3
CATALOG_TYPE = "wam_artifact_cache"
ARTIFACT_TYPE = "wam_artifact"
TENSOR_SHARD_BACKEND = f"tensor_shard_v{FORMAT_VERSION}"


def revision_tag(name: str, revision: int) -> str:
    """Build a stable internal revision token without exposing product labels."""

    return f"{str(name)}_v{int(revision)}"
