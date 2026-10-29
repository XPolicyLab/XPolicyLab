from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any


def _integer(value: Any, key: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, str)):
        raise ValueError(f"{key} must be an integer >= {minimum}")
    try:
        parsed = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{key} must be an integer >= {minimum}") from error
    if parsed < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return parsed


@dataclass(frozen=True)
class TTCConfig:
    enabled: bool = False
    num_proposals: int = 4
    timestep_ratios: tuple[float, ...] = (0.5,)
    noise_seed: int | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TTCConfig":
        enabled = config.get("ttc_enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("ttc_enabled must be a YAML boolean: true or false")
        proposals = _integer(config.get("rcs_num_proposals", 4), "rcs_num_proposals", 1)
        raw_ratios = config.get("rcs_timestep_ratios", [0.5])
        if isinstance(raw_ratios, str):
            raw_ratios = raw_ratios.split(",")
        try:
            ratios = tuple(float(r) for r in raw_ratios)
        except (ValueError, TypeError) as error:
            raise ValueError("rcs_timestep_ratios must be a nonempty list in [0, 1)") from error
        if not ratios or any(not math.isfinite(r) or not 0 <= r < 1 for r in ratios):
            raise ValueError("rcs_timestep_ratios must be a nonempty list in [0, 1)")
        seed = config.get("rcs_noise_seed")
        if seed is not None:
            seed = _integer(seed, "rcs_noise_seed", 0)
            if seed >= 2**63:
                raise ValueError("rcs_noise_seed must be less than 2**63")
        return cls(enabled, proposals, ratios, seed)

    def metadata(self) -> dict[str, Any]:
        return {
            "ttc_enabled": self.enabled,
            "ttc_method": "rcs_best_of_n" if self.enabled else None,
            "rcs_num_proposals": self.num_proposals,
            "rcs_timestep_ratios": list(self.timestep_ratios),
            "rcs_noise_seed": self.noise_seed,
        }

    def validate_backend(self, metadata: Mapping[str, Any]) -> None:
        actual_enabled = metadata.get("ttc_enabled", False)
        if actual_enabled is not self.enabled:
            raise ValueError("TTC enabled mismatch: restart proxy and backend with the same deploy.yml")
        if not self.enabled:
            return
        actual = TTCConfig.from_mapping(metadata)
        if actual != self or metadata.get("ttc_method") != "rcs_best_of_n":
            raise ValueError("TTC parameters mismatch: restart proxy and backend with the same deploy.yml")
