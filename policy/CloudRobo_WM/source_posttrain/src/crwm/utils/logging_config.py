import logging
from typing import Optional
import os
import torch.distributed as dist
from rich.logging import RichHandler
_compact_logging = False

class _TrainingLogFilter(logging.Filter):

    def __init__(self, main_process):
        super().__init__()
        self.main_process = main_process

    def filter(self, record):
        if record.levelno >= logging.WARNING and record.getMessage().startswith('Removed shared tensor {'):
            record.msg = 'Checkpoint serialization omitted shared parameter aliases; validate them on restore.'
            record.args = ()
        if record.levelno >= logging.WARNING:
            return True
        if not self.main_process or record.levelno < logging.INFO:
            return False
        return record.name in {'crwm.trainer', 'crwm.runtime'} or (record.name == 'crwm.models.action_dit' and record.getMessage().startswith('Loaded ActionDiT backbone'))

def setup_logging(log_level: int=logging.INFO, is_main_process: Optional[bool]=None, rich_handler_kwargs: Optional[dict]=None, formatter_kwargs: Optional[dict]=None, preserve_hydra_handlers: bool=True, compact: bool=False) -> None:
    global _compact_logging
    _compact_logging = compact
    if is_main_process is None:
        is_main_process = _is_main_process()
    root_logger = logging.getLogger()
    if compact:
        retained = []
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if is_main_process and preserve_hydra_handlers:
                    retained.append(handler)
                else:
                    handler.close()
        root_logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        for handler in [logging.StreamHandler(), *retained]:
            handler.setFormatter(formatter)
            handler.addFilter(_TrainingLogFilter(is_main_process))
            root_logger.addHandler(handler)
        root_logger.setLevel(log_level)
        for name, child in logging.root.manager.loggerDict.items():
            if name.startswith('crwm.') and isinstance(child, logging.Logger):
                child.disabled = False
                child.propagate = True
        return
    if is_main_process:
        existing_file_handlers = []
        if preserve_hydra_handlers:
            existing_file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        root_logger.handlers.clear()
        default_rich_kwargs = {'markup': True, 'rich_tracebacks': True, 'show_level': True, 'show_path': True, 'show_time': True}
        if rich_handler_kwargs:
            default_rich_kwargs.update(rich_handler_kwargs)
        rich_handler = RichHandler(**default_rich_kwargs)
        default_formatter_kwargs = {'fmt': '| >> %(message)s', 'datefmt': '%m/%d [%H:%M:%S]'}
        if formatter_kwargs:
            default_formatter_kwargs.update(formatter_kwargs)
        formatter = logging.Formatter(**default_formatter_kwargs)
        rich_handler.setFormatter(formatter)
        root_logger.addHandler(rich_handler)
        for handler in existing_file_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(log_level)
    else:
        root_logger.setLevel(logging.ERROR)

def _is_main_process() -> bool:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    for key in ('RANK', 'SLURM_PROCID', 'LOCAL_RANK'):
        if key in os.environ:
            return os.environ.get(key, '0') in ('0', '0\n', '')
    return True

def get_logger(name: str=__name__, level: int=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if _compact_logging:
        logger.propagate = True
        logger.disabled = False
    elif not _is_main_process():
        logger.propagate = False
        logger.disabled = True
    return logger
