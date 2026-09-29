"""Dataset mixture aliases.

Upstream layered a JointFlow-local alias table on top of a globally mutated
registry that other benchmarks wrote into at import time, plus derived
RoboTwin clean/randomized aliases.  This release resolves exactly one mixture,
so the lookup is a direct read of the RoboDojo registry.
"""

from __future__ import annotations

from cogwam.data.robodojo import DATASET_NAMED_MIXTURES


def resolve_data_mix(data_mix: str) -> list[tuple[str, float, str]]:
    if data_mix in DATASET_NAMED_MIXTURES:
        return DATASET_NAMED_MIXTURES[data_mix]
    known = sorted(DATASET_NAMED_MIXTURES)
    raise KeyError(f"Unknown data_mix={data_mix!r}. Known mixtures include: {known}")


__all__ = ["DATASET_NAMED_MIXTURES", "resolve_data_mix"]
