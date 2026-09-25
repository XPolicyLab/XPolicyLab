import logging
from typing import Optional
import os

import torch.distributed as dist

from rich.console import Console
from rich.text import Text


class _LowercaseLevelFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        record.levelname = original_levelname.lower()
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


class _NoisyCompilerCommandFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _env_flag("WAM_VERBOSE_THIRD_PARTY_LOGS"):
            return True
        message = record.getMessage()
        if message.startswith("PyTorch version ") and message.endswith(" available."):
            return False
        return not (
            message.startswith("gcc ")
            and "/wam_tmp/" in message
            and ("test.o" in message or "test.c" in message or "a.out" in message)
        )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


class _CompactRichHandler(logging.Handler):
    _LEVEL_STYLES = {
        "debug": "cyan",
        "info": "green",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }

    def __init__(
        self,
        *,
        datefmt: str = "%m/%d [%H:%M:%S]",
        force_terminal: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if force_terminal is None:
            force_terminal = False if "NO_COLOR" in os.environ else True
        self.datefmt = datefmt
        self.console = Console(
            force_terminal=force_terminal,
            color_system="auto",
            no_color=True if "NO_COLOR" in os.environ else None,
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            level = record.levelname.lower()
            rendered = Text()
            rendered.append(self.formatter.formatTime(record, self.datefmt), style="dim")
            rendered.append(" ")
            rendered.append(f"[{level}]", style=self._LEVEL_STYLES.get(level, "white"))
            rendered.append(" ")
            rendered.append(message)
            if record.exc_info:
                rendered.append("\n")
                rendered.append(self.formatter.formatException(record.exc_info), style="red")
            if record.stack_info:
                rendered.append("\n")
                rendered.append(str(record.stack_info), style="dim")
            self.console.print(rendered, soft_wrap=True)
        except Exception:
            self.handleError(record)


def _silence_noisy_third_party_loggers() -> None:
    if _env_flag("WAM_VERBOSE_THIRD_PARTY_LOGS"):
        return
    for name in (
        "torch.utils.cpp_extension",
        "torch._inductor",
        "deepspeed",
        "deepspeed.ops",
        "deepspeed.ops.op_builder",
        "deepspeed.runtime",
        "triton",
    ):
        noisy_logger = logging.getLogger(name)
        noisy_logger.setLevel(logging.WARNING)


def setup_logging(
    log_level: int = logging.INFO,
    is_main_process: Optional[bool] = None,
    rich_handler_kwargs: Optional[dict] = None,
    formatter_kwargs: Optional[dict] = None,
    preserve_hydra_handlers: bool = True
) -> None:
    """
    Configure the logging system for the entire codebase.

    In distributed training, only the main process outputs logs while other processes are silenced.
    This function configures the root logger so all child loggers inherit the same configuration.

    Args:
        log_level: Logging level (default INFO), only applies to main process
        is_main_process: Whether this is the main process. If None, infer automatically.
        rich_handler_kwargs: Deprecated; kept for compatibility.
        formatter_kwargs: Additional kwargs to pass to Formatter (fmt and datefmt)
        preserve_hydra_handlers: Keep existing FileHandlers from Hydra (default True)

    Example:
        ```python
        # In a single-machine script
        from wam.utils.logging_config import setup_logging
        setup_logging()

        # In a distributed training script
        from accelerate import PartialState
        from wam.utils.logging_config import setup_logging

        distributed_state = PartialState()
        setup_logging(
            log_level=logging.INFO,
            is_main_process=distributed_state.is_main_process
        )
        ```
    """
    if is_main_process is None:
        is_main_process = _is_main_process()

    root_logger = logging.getLogger()

    if is_main_process:
        # Save existing FileHandlers (e.g., from Hydra) if requested
        existing_file_handlers = []
        if preserve_hydra_handlers:
            existing_file_handlers = [
                h for h in root_logger.handlers
                if isinstance(h, logging.FileHandler)
            ]

        # Clear all default handlers on the root logger
        root_logger.handlers.clear()

        default_formatter_kwargs = {
            "fmt": "%(asctime)s [%(levelname)s] %(message)s",
            "datefmt": "%m/%d [%H:%M:%S]",
        }
        if formatter_kwargs:
            default_formatter_kwargs.update(formatter_kwargs)

        rich_kwargs = dict(rich_handler_kwargs or {})
        console_handler = _CompactRichHandler(
            datefmt=str(default_formatter_kwargs.get("datefmt", "%m/%d [%H:%M:%S]")),
            force_terminal=rich_kwargs.pop("force_terminal", None),
        )
        console_handler.setFormatter(_LowercaseLevelFormatter(**default_formatter_kwargs))
        console_handler.addFilter(_NoisyCompilerCommandFilter())

        # Add handler and set logging level
        root_logger.addHandler(console_handler)

        # Restore existing FileHandlers from Hydra
        for handler in existing_file_handlers:
            handler.addFilter(_NoisyCompilerCommandFilter())
            root_logger.addHandler(handler)

        root_logger.setLevel(log_level)

    else:
        # In non-main processes, set root logger level to ERROR to silence all logs
        root_logger.setLevel(logging.ERROR)

    _silence_noisy_third_party_loggers()


def _is_main_process() -> bool:
    """
    Best-effort check for main process without any synchronization.
    """
    # Prefer torch.distributed state if initialized.
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    # Fallback to environment variables commonly set by launchers.
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0") in ("0", "0\n", "")

    return True

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Drop-in replacement for accelerate.logging.get_logger:
    - No implicit barriers.
    - Only the main process emits log records by default.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # if not logger.handlers and _is_main_process():
    #     handler = logging.StreamHandler()
    #     formatter = logging.Formatter(
    #         fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #         datefmt="%Y-%m-%d %H:%M:%S",
    #     )
    #     handler.setFormatter(formatter)
    #     logger.addHandler(handler)

    if not _is_main_process():
        logger.propagate = False
        logger.disabled = True

    return logger
