"""Abstract contract implemented by inference policies."""

from abc import ABC
from abc import abstractmethod
from typing import Any

from soma.inference.types import ImageInput


class BasePolicy(ABC):
    """Minimal lifecycle contract for one embodied policy."""

    @abstractmethod
    def infer(
        self,
        images: ImageInput,
        instruction: str,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one policy inference request."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset episode-scoped policy state."""
        raise NotImplementedError
