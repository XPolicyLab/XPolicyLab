# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeAlias,
    runtime_checkable,
)

import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType,
)
from robo_orchard_core.utils.hook import (
    HookContext,
    HookContextChannel,
    RemoveableHandle,
)
from robo_orchard_core.utils.string import add_indentation
from typing_extensions import Self, TypeVar

__all__ = [
    "ClassType",
    "HookContext",
    "PipelineHooks",
    "PipelineHookArgs",
    "PipelineHookChanelType",
    "PipelineHooksConfig",
    "ModelOutput",
    "ModelOutputHasLossKeys",
]


@runtime_checkable
class ModelOutput(Protocol):
    """A protocol representing a model output that contains key-value pairs."""

    def __getitem__(self, key: str) -> Any: ...

    def keys(self) -> Iterable[Any]: ...

    def items(self) -> tuple[Any, Any]: ...

    def values(self) -> Iterable[Any]: ...


@runtime_checkable
class ModelOutputHasLossKeys(ModelOutput, Protocol):
    def loss_keys(self) -> Iterable[str]: ...


@dataclass
class PipelineHookArgs:
    """A data class for passing arguments to hook functions.

    This class serves as a container for various parameters and state
    information required by hooks at different stages of the training or
    evaluation pipeline. It is designed to be flexible and extensible for
    different training configurations.
    """

    accelerator: Accelerator
    epoch_id: int = 0
    step_id: int = 0
    global_step_id: int = 0
    max_epoch: Optional[int] = None
    max_step: Optional[int] = None
    start_epoch: int = 0
    start_step: int = 0
    dataloader: Optional[Iterable] = None
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[AcceleratedOptimizer] = None
    lr_scheduler: Optional[AcceleratedScheduler] = None
    batch: Optional[Any] = None
    model_outputs: Optional[ModelOutput] = None
    reduce_loss: Optional[torch.Tensor] = None
    """The average loss across all processes, if applicable.

    If the model output loss, the loss will be reduced (averaged)
    across all processes after the forward pass. So the value is
    different from the loss returned by the model, which is the
    loss for the current process only for backward computation.
    """

    def copy_with_updates(self, **kwargs):
        """Create a copy of the current instance with updated attributes.

        This method allows you to create a new instance of the class with
        modified attributes while keeping the original instance unchanged.

        Args:
            **kwargs: Keyword arguments representing the attributes to be
                updated. The keys should match the attribute names of the
                class.

        Returns:
            PipelineHookArgs: A new instance of the class with updated
                attributes.
        """

        instance = self.__class__(**self.__dict__)
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute")
        return instance


PipelineHookChanelType: TypeAlias = Literal[
    "on_loop",  # the whole training loop pipeline
    "on_epoch",  # in one epoch pipeline
    "on_step",  # in one step pipeline.
    "on_batch",  # in one batch pipeline
    "on_model_forward",  # only in model forward pipeline
    "on_model_backward",  # only in model backward pipeline
]


class PipelineHooks(ClassInitFromConfigMixin):
    """A class to manage pipeline hooks for training processes.

    This class only accept config class as input for the constructor.

    """

    def __init__(self):
        self.hooks: dict[
            PipelineHookChanelType, HookContextChannel[PipelineHookArgs]
        ] = {}
        for c in PipelineHookChanelType.__args__:
            self.hooks[c] = HookContextChannel[PipelineHookArgs](c)

    @contextmanager
    def begin(self, channel: PipelineHookChanelType, arg: PipelineHookArgs):
        with self.hooks[channel].begin(arg) as ctx:
            yield ctx

    def register_hook(
        self,
        channel: PipelineHookChanelType,
        hook: HookContext[PipelineHookArgs],
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a hook context handler.

        Args:
            channel (PipelineHookChanelType): The channel to register the hook.
            hook (HookContext[PipelineHookArgs]): The hook context handler
                to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hook.
        """
        return self.hooks[channel].register(hook)

    def register_pipeline_hooks(
        self,
        hooks: PipelineHooks,
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a set of pipeline hooks.

        Args:
            hooks (PipelineHooks[T]): The pipeline hooks to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hooks.
        """
        handles: list[RemoveableHandle] = []
        for channel, hook in hooks.hooks.items():
            handles.append(self.hooks[channel].register_hook_channel(hook))

        def remove():
            for handle in handles:
                handle()

        return RemoveableHandle(remove)

    def __iadd__(self, other: PipelineHooks) -> Self:
        """Add another set of pipeline hooks to the current instance.

        Args:
            other (PipelineHooks): The other set of pipeline hooks to add.

        Returns:
            PipelineHooks: The updated instance with the added hooks.
        """
        self.register_pipeline_hooks(other)
        return self

    def unregister_all(self):
        """Unregister all hook context handlers."""
        for channel in self.hooks.values():
            channel.unregister_all()

    @classmethod
    def from_hooks(
        cls: Type[Self],
        hooks: (
            Self
            | PipelineHooksConfig
            | Iterable[Self | PipelineHooksConfig]
            | None
        ),
    ) -> Self:
        """Create a new instance of the class from a list of hooks.

        Args:
            hooks (Self | Iterable[Self] | None): A list of hooks to register.

        Returns:
            Self: A new instance of the class with the registered hooks.
        """

        if hooks is None:
            return cls()

        if isinstance(hooks, (PipelineHooksConfig, PipelineHooks)):
            hooks_or_cfg_: list[PipelineHooks | PipelineHooksConfig] = [hooks]
        else:
            hooks_or_cfg_ = hooks  # type: ignore

        input_hooks: list[PipelineHooks] = []
        for hook_or_cfg in hooks_or_cfg_:
            if isinstance(hook_or_cfg, PipelineHooksConfig):
                hook = hook_or_cfg()
            elif isinstance(hook_or_cfg, PipelineHooks):
                hook = hook_or_cfg
            else:
                raise TypeError(
                    f"Expected PipelineHooks or PipelineHooksConfig, "
                    f"but got {type(hook_or_cfg)}"
                )
            input_hooks.append(hook)

        ret = cls()
        for hook in input_hooks:
            ret += hook
        return ret

    def __repr__(self) -> str:
        hook_str = "{"
        for k, v in self.hooks.items():
            if len(v) == 0:
                continue
            hook_str += "\n" + add_indentation(f"{k}: {v}, ", indent=2)
        if hook_str != "{":
            hook_str += "\n"
        hook_str += "}"
        content = f"hooks={hook_str}"
        ret = (
            f"<{self.__class__.__module__}.{self.__class__.__name__}(\n"
            + add_indentation(content, indent=2, first_line_indent=True)
            + ")>"
        )
        return ret


PipelineHooksType_co = TypeVar(
    "PipelineHooksType_co",
    bound=PipelineHooks,
    covariant=True,
)


class PipelineHooksConfig(ClassConfig[PipelineHooksType_co]):
    pass
