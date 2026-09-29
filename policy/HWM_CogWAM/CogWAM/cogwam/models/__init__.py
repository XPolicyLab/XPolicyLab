"""Model components.

Imports are lazy: loading ``cogwam.models.cogwam`` pulls in transformers and the
DINO backbone, which tooling that only needs the recipe contract should not pay
for.
"""

from cogwam.models.base import FRAMEWORK_NAME, CogWAMBase, build_model, load_state_dict_file

__all__ = ["FRAMEWORK_NAME", "CogWAM", "CogWAMBase", "build_model", "load_state_dict_file"]


def __getattr__(name: str):
    if name == "CogWAM":
        from cogwam.models.cogwam import CogWAM

        return CogWAM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
