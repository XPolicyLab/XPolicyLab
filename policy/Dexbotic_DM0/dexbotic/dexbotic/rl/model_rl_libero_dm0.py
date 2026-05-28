"""RL training entry: Libero + Dexbotic-DM0 + PPO (cfg in ``dexbotic/config``).

Usage
-----
Default run (libero_spatial suite)::

    python -m dexbotic.rl.model_rl_libero_dm0

Pick a different Libero suite via ``--suite=<name>``. Valid suites are
``libero_10``, ``libero_90``, ``libero_goal``, ``libero_object``, ``libero_spatial``::

    python -m dexbotic.rl.model_rl_libero_dm0 --suite=libero_10

Power users can still pass any Hydra config by name directly
(all suite configs live in ``dexbotic/config/rl/libero_*_ppo_dexbotic_dm0.yaml``)::

    python -m dexbotic.rl.model_rl_libero_dm0 --config-name=libero_goal_ppo_dexbotic_dm0
"""

from __future__ import annotations

import sys

from dexbotic.rl.rlinf_registry import register_all

register_all()

import hydra

from dexbotic.rl._embodied_cli import run_embodied_rl

_MODEL_TAG = "dexbotic_dm0"
_SUPPORTED_SUITES = (
    "libero_10",
    "libero_90",
    "libero_goal",
    "libero_object",
    "libero_spatial",
)
_DEFAULT_SUITE = "libero_spatial"


def _resolve_config_name_from_argv() -> str:
    """Strip ``--suite=<name>`` from ``sys.argv`` and return the matching config name.

    If ``--config-name`` is already supplied (Hydra native), leave argv alone and return
    the default (Hydra will override it anyway).
    """
    suite = _DEFAULT_SUITE
    remaining: list[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("--suite="):
            suite = arg.split("=", 1)[1]
            continue
        remaining.append(arg)
    sys.argv[1:] = remaining

    if suite not in _SUPPORTED_SUITES:
        raise SystemExit(
            f"Unknown --suite={suite!r}. Valid: {list(_SUPPORTED_SUITES)}"
        )
    return f"{suite}_ppo_{_MODEL_TAG}"


_CONFIG_NAME = _resolve_config_name_from_argv()


@hydra.main(
    version_base="1.1",
    config_path="../config/rl",
    config_name=_CONFIG_NAME,
)
def main(cfg) -> None:
    run_embodied_rl(cfg)


if __name__ == "__main__":
    main()
