from __future__ import annotations

from typing import Any

from evals.common.simulator import (
    TaskFactoryEvaluationEnvironment,
    load_optional_factory,
)


class LiberoTaskSuiteEnvironment(TaskFactoryEvaluationEnvironment):
    """Create one LIBERO BDDL task environment per evaluation trial."""

    def __init__(
        self,
        *,
        module_name: str = "libero.libero.envs",
        factory_name: str = "OffScreenRenderEnv",
        environment_args: tuple[Any, ...] = (),
        environment_kwargs: dict[str, Any] | None = None,
        metadata_to_factory: dict[str, str] | None = None,
        capture_frames: bool = False,
    ) -> None:
        super().__init__(
            module_name=module_name,
            factory_name=factory_name,
            backend_name="LIBERO",
            environment_args=environment_args,
            environment_kwargs=environment_kwargs,
            metadata_to_factory=(
                {"bddl_file_name": "bddl_file_name"}
                if metadata_to_factory is None
                else metadata_to_factory
            ),
            capture_frames=capture_frames,
        )


def make_libero_environment(
    *args: Any,
    module_name: str = "libero.libero.envs",
    factory_name: str = "OffScreenRenderEnv",
    **kwargs: Any,
) -> Any:
    """Construct LIBERO without importing it at adapter import time."""

    factory = load_optional_factory(
        module_name,
        factory_name,
        backend_name="LIBERO",
    )
    return factory(*args, **kwargs)


__all__ = ["LiberoTaskSuiteEnvironment", "make_libero_environment"]
