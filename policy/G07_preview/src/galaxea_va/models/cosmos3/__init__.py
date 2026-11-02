"""Cosmos3(-Nano) backbone vendored into fastwam.

This package re-implements NVIDIA's Cosmos3 Omni diffusion transformer as a
self-contained set of ``torch.nn`` modules (no ``diffusers`` dependency), driven
by a fastwam-style batched forward + dense asymmetric MoT attention mask so it
can replace the Wan2.2 base model in the FastWAM video+action policy.

The base (uncond-equivalent) model mirrors ``galaxea_va.models.wan22.fastwam.FastWAM``;
the joint / idm / optional-idm / AR variants mirror the corresponding wan22
subclasses.

Classes are exported lazily so that importing the lightweight submodules
(``transformer``/``masks``/``mrope``) does not pull in the heavier VAE/metrics chain.
"""

__all__ = [
    "Cosmos3FastWAM",
    "Cosmos3FastWAMJoint",
    "Cosmos3FastWAMIDM",
    "Cosmos3FastWAMOptionalIDM",
    "Cosmos3FastWAMAR",
    "Cosmos3FastWAMReferenceAR",
]

_LAZY = {
    "Cosmos3FastWAM": ".cosmos3_fastwam",
    "Cosmos3FastWAMJoint": ".cosmos3_fastwam_joint",
    "Cosmos3FastWAMIDM": ".cosmos3_fastwam_idm",
    "Cosmos3FastWAMOptionalIDM": ".cosmos3_fastwam_optional_idm",
    "Cosmos3FastWAMAR": ".cosmos3_fastwam_ar",
    "Cosmos3FastWAMReferenceAR": ".cosmos3_fastwam_reference_ar",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is not None:
        import importlib

        return getattr(importlib.import_module(module, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
