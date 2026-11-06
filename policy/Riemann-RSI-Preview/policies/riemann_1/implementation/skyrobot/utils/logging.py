import logging
import os
import sys


logger = logging.getLogger()

def init_logger() -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[skyrobot-wam] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


def _is_rank0() -> bool:
    import torch.distributed as dist
    return not dist.is_initialized() or dist.get_rank() == 0


def rank0_info(msg: str, *args, **kwargs) -> None:
    if _is_rank0():
        logger.info(msg, *args, **kwargs)


def rank0_warning(msg: str, *args, **kwargs) -> None:
    if _is_rank0():
        logger.warning(msg, *args, **kwargs)


_logged: set[str] = set()


def warn_once(logger: logging.Logger, msg: str) -> None:
    """Log a warning message only once per unique message.

    Uses a global set to track messages that have already been logged
    to prevent duplicate warning messages from cluttering the output.

    Args:
        logger (logging.Logger): The logger instance to use for warning.
        msg (str): The warning message to log.
    """
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)
