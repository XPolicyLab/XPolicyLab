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
import logging
import os
import shutil
from typing import Optional, TypeVar

from accelerate import Accelerator

from robo_orchard_lab.models.torch_model import TorchModelMixin
from robo_orchard_lab.pipeline.hooks.mixin import (
    ClassType,
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = [
    "SaveCheckpoint",
    "SaveCheckpointConfig",
    "SaveModel",
    "SaveModelConfig",
]


SaveHookType = TypeVar("SaveHookType", bound=PipelineHooks)

logger = logging.getLogger(__name__)


class SaveHookConfigT(PipelineHooksConfig[SaveHookType]):
    save_root: str | None = None
    """The root directory where checkpoints are saved."""
    save_epoch_freq: Optional[int] = 1
    """Frequency of saving checkpoints based on epochs."""
    save_step_freq: Optional[int] = None
    """Frequency of saving checkpoints based on steps."""
    save_when_loop_end: bool = True
    """Whether to save the model at the end of the training loop."""
    save_model: bool = True
    """Whether to save the model as well."""

    def __post_init__(self):
        if self.save_epoch_freq is None and self.save_step_freq is None:
            raise ValueError(
                "Either `save_epoch_freq` or `save_step_freq` must be "
                "specified. Both cannot be None."
            )
        if self.save_epoch_freq is not None and self.save_epoch_freq < 1:
            raise ValueError(
                "save_epoch_freq = {} < 1 is not allowed".format(
                    self.save_epoch_freq
                )
            )
        if self.save_step_freq is not None and self.save_step_freq < 1:
            raise ValueError(
                "save_step_freq = {} < 1 is not allowed".format(
                    self.save_step_freq
                )
            )


class SaveCheckpoint(PipelineHooks):
    """Checkpoint hook.

    A checkpointing hook for saving model state at specified epoch or
    step intervals.


    Args:
        cfg (SaveCheckpointConfig): Configuration object containing
            parameters for checkpointing.

    """

    def __init__(
        self,
        cfg: SaveCheckpointConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_root = cfg.save_root
        self.save_epoch_freq = cfg.save_epoch_freq
        self.save_step_freq = cfg.save_step_freq
        self._is_checked = False

        self.register_hook(
            channel="on_step",
            hook=HookContext.from_callable(after=self._on_step_end),
        )
        self.register_hook(
            channel="on_epoch",
            hook=HookContext.from_callable(after=self._on_epoch_end),
        )
        self.register_hook(
            channel="on_loop",
            hook=HookContext.from_callable(after=self._on_loop_end),
        )

    def _check(self, accelerator: Accelerator) -> None:
        if self._is_checked:
            return
        if (
            not accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is None
        ):
            raise ValueError(
                "save_root is required when `automatic_checkpoint_naming of Accelerator is False"  # noqa: E501
            )
        if (
            accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is not None
        ):
            logger.warning(
                "Ignore customize save_root = {} since `automatic_checkpoint_naming` is True for Accelerator".format(  # noqa: E501
                    self.save_root
                )
            )
        self._is_checked = True

    def _save_state(self, accelerator: Accelerator) -> str:
        accelerator.wait_for_everyone()
        save_location = accelerator.save_state(self.save_root)  # type: ignore
        if self.cfg.save_model and accelerator.is_main_process:
            for idx, model in enumerate(accelerator._models):
                model_prefix = f"model_{idx}"
                if idx == 0 and len(accelerator._models) == 1:
                    model_prefix = "model"
                save_directory = os.path.join(save_location, model_prefix)
                accelerator.save_model(model, save_directory)
                model_config_path = os.path.join(
                    save_location, f"{model_prefix}.config.json"
                )
                if os.path.exists(model_config_path):
                    # copy config to save_directory
                    shutil.copy(model_config_path, save_directory)

        return save_location

    def _on_step_end(self, args: PipelineHookArgs) -> None:
        """Callback when step ends.

        Saves a checkpoint at the end of a step if `save_step_freq` conditions
        are met.


        Args:
            args (HookArgs): Arguments containing the `global_step_id` and
                `accelerator` to facilitate saving state.

        Logs:
            A message indicating the checkpoint location.
        """
        if (
            self.save_step_freq is not None
            and (args.global_step_id + 1) % self.save_step_freq == 0
        ):
            self._check(args.accelerator)
            save_location = self._save_state(args.accelerator)
            if args.accelerator.is_main_process:
                logger.info(
                    "Save checkpoint at the end of step {} to {}".format(
                        args.global_step_id, save_location
                    )
                )

    def _on_epoch_end(self, args: PipelineHookArgs) -> None:
        """Callback when epoch ends.

        Saves a checkpoint at the end of an epoch if `save_epoch_freq`
        conditions are met.

        Args:
            args (HookArgs): Arguments containing the `epoch_id` and
                `accelerator` to facilitate saving state.

        Logs:
            A message indicating the checkpoint location.
        """
        if (
            self.save_epoch_freq is not None
            and (args.epoch_id + 1) % self.save_epoch_freq == 0
        ):
            self._check(args.accelerator)
            save_location = self._save_state(args.accelerator)
            if args.accelerator.is_main_process:
                logger.info(
                    "Save checkpoint at the end of epoch {} to {}".format(
                        args.epoch_id, save_location
                    )
                )

    def _on_loop_end(self, args: PipelineHookArgs) -> None:
        if not self.cfg.save_when_loop_end:
            return

        self._check(args.accelerator)
        save_location = self._save_state(args.accelerator)
        if args.accelerator.is_main_process:
            logger.info(
                "Save checkpoint at the end of training to {}".format(
                    save_location
                )
            )


class SaveCheckpointConfig(SaveHookConfigT[SaveCheckpoint]):
    """Configuration class for SaveCheckpoint."""

    class_type: ClassType[SaveCheckpoint] = SaveCheckpoint


class SaveModel(PipelineHooks):
    cfg: SaveModelConfig

    def __init__(
        self,
        cfg: SaveModelConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.register_hook(
            channel="on_step",
            hook=HookContext.from_callable(after=self._on_step_end),
        )
        self.register_hook(
            channel="on_epoch",
            hook=HookContext.from_callable(after=self._on_epoch_end),
        )

        self.register_hook(
            channel="on_loop",
            hook=HookContext.from_callable(after=self._on_loop_end),
        )
        # use to avoid duplicate saving at epoch end or loop end
        self._saved_last_step: int = -1

    def _save_model(
        self, args: PipelineHookArgs, step_id: int, epoch_id: int
    ) -> str:
        """Saves the model state using the provided accelerator.

        Args:
            args (PipelineHookArgs): The arguments containing the
                accelerator and other relevant information.

        Raises:
            ValueError: If the save root is not specified when
                automatic checkpoint naming is disabled.
        """
        if args.model is None:
            raise ValueError("model is None. Cannot save model.")
        self._saved_last_step = args.global_step_id
        accelerator = args.accelerator
        save_root = self.cfg.get_save_root(accelerator)
        save_root = os.path.join(
            save_root, f"model_epoch_{epoch_id}_step_{step_id}"
        )
        if not accelerator.is_main_process:
            return save_root
        # only save on main process
        for idx, model in enumerate(accelerator._models):
            save_directory = os.path.join(save_root, f"{idx}")
            accelerator.save_model(model, save_directory)
            if isinstance(model, TorchModelMixin):
                model_config_path = os.path.join(
                    save_directory, "model.config.json"
                )
                with open(model_config_path, "w") as f:
                    f.write(model.cfg.model_dump_json(indent=4))

        return save_root

    def _on_step_end(self, args: PipelineHookArgs) -> None:
        if (
            self.cfg.save_step_freq is not None
            and (args.global_step_id + 1) % self.cfg.save_step_freq == 0
            and self._saved_last_step != args.global_step_id
        ):
            args.accelerator.wait_for_everyone()
            if args.accelerator.is_main_process:
                save_location = self._save_model(
                    args, step_id=args.step_id, epoch_id=args.epoch_id
                )
                logger.info(
                    "Save model at the end of step {} to {}".format(
                        args.global_step_id, save_location
                    )
                )

    def _on_epoch_end(self, args: PipelineHookArgs) -> None:
        if (
            self.cfg.save_epoch_freq is not None
            and (args.epoch_id + 1) % self.cfg.save_epoch_freq == 0
            and self._saved_last_step != args.global_step_id
        ):
            args.accelerator.wait_for_everyone()
            if args.accelerator.is_main_process:
                save_location = self._save_model(
                    args, step_id=args.step_id, epoch_id=args.epoch_id
                )
                logger.info(
                    "Save model at the end of epoch {} of global step {} to {}".format(  # noqa: E501
                        args.epoch_id, args.global_step_id, save_location
                    )
                )

    def _on_loop_end(self, args: PipelineHookArgs) -> None:
        """Saves the model at the end of the training loop.

        This method ensures that the model is saved at the end of training,
        even if the last epoch or step did not meet the saving criteria
        defined by `save_epoch_freq` or `save_step_freq`.

        Args:
            args (PipelineHookArgs): The arguments containing the
                accelerator and other relevant information.

        Logs:
            A message indicating the checkpoint location.
        """
        if (
            self.cfg.save_when_loop_end is False
            or self._saved_last_step == args.global_step_id
        ):
            return

        args.accelerator.wait_for_everyone()

        if args.accelerator.is_main_process:
            save_location = self._save_model(
                args, step_id=args.step_id, epoch_id=args.epoch_id
            )
            logger.info(
                "Save model at the end of training global step {} to {}".format(  # noqa: E501
                    args.global_step_id, save_location
                )
            )


class SaveModelConfig(SaveHookConfigT[SaveModel]):
    class_type: ClassType[SaveModel] = SaveModel

    def get_save_root(self, accelerator: Accelerator) -> str:
        """Get the root directory for saving checkpoints.

        If `automatic_checkpoint_naming` is enabled in the Accelerator's
        project configuration, the save root is set to a default directory
        within the project's directory. Otherwise, it uses the provided
        `save_root` value.
        """
        if (
            not accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is None
        ):
            raise ValueError(
                "save_root is required when `automatic_checkpoint_naming of Accelerator is False"  # noqa: E501
            )
        if (
            accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is not None
        ):
            logger.warning(
                "Ignore customize save_root = {} since `automatic_checkpoint_naming` is True for Accelerator".format(  # noqa: E501
                    self.save_root
                )
            )
        if accelerator.project_configuration.automatic_checkpoint_naming:
            return os.path.join(accelerator.project_dir, "saved_models")
        else:
            if self.save_root is None:
                raise ValueError(
                    "save_root must be set when automatic naming is off."
                )
            return self.save_root
