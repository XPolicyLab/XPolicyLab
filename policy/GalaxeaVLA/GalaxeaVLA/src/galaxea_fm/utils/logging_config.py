import logging
from typing import Optional

from accelerate import Accelerator
from rich.logging import RichHandler

# Configure unified logging system (applies to all modules in the codebase)
# Hydra's FileHandler (configured in hydra.yaml) will be preserved automatically


def setup_logging(
    log_level: int = logging.INFO,
    is_main_process: bool = True,
    rich_handler_kwargs: Optional[dict] = None,
    formatter_kwargs: Optional[dict] = None,
    preserve_hydra_handlers: bool = True
) -> None:
    """
    Configure the logging system for the entire codebase with Rich formatting.
    
    In distributed training, only the main process outputs logs while other processes are silenced.
    This function configures the root logger so all child loggers inherit the same configuration.
    
    Args:
        log_level: Logging level (default INFO), only applies to main process
        is_main_process: Whether this is the main process (default True)
        rich_handler_kwargs: Additional kwargs to pass to RichHandler
        formatter_kwargs: Additional kwargs to pass to Formatter (fmt and datefmt)
        preserve_hydra_handlers: Keep existing FileHandlers from Hydra (default True)

    Example:
        ```python
        # In a single-machine script
        from galaxea_fm.utils.logging_config import setup_logging
        setup_logging()
        
        # In a distributed training script
        from accelerate import PartialState
        from galaxea_fm.utils.logging_config import setup_logging
        
        distributed_state = PartialState()
        setup_logging(
            log_level=logging.INFO,
            is_main_process=distributed_state.is_main_process
        )
        ```
    """
    root_logger = logging.getLogger()
    
    # Save existing FileHandlers (e.g., from Hydra) if requested
    existing_file_handlers = []
    if preserve_hydra_handlers:
        existing_file_handlers = [
            h for h in root_logger.handlers 
            if isinstance(h, logging.FileHandler)
        ]
    
    # Clear all default handlers on the root logger
    root_logger.handlers.clear()
    
    # Configure RichHandler parameters
    default_rich_kwargs = {
        "markup": True,
        "rich_tracebacks": True,
        "show_level": True,
        "show_path": True,
        "show_time": True,
    }
    if rich_handler_kwargs:
        default_rich_kwargs.update(rich_handler_kwargs)
    
    # Create RichHandler
    rich_handler = RichHandler(**default_rich_kwargs)
    
    # Configure Formatter
    default_formatter_kwargs = {
        "fmt": "| >> %(message)s",
        "datefmt": "%m/%d [%H:%M:%S]",
    }
    if formatter_kwargs:
        default_formatter_kwargs.update(formatter_kwargs)
    
    formatter = logging.Formatter(**default_formatter_kwargs)
    rich_handler.setFormatter(formatter)
    
    # Add handler and set logging level
    root_logger.addHandler(rich_handler)

    # Restore existing FileHandlers from Hydra
    for handler in existing_file_handlers:
        root_logger.addHandler(handler)

    if is_main_process:
        root_logger.setLevel(log_level)
        
    else:
        # In non-main processes, set root logger level to ERROR to silence all logs
        root_logger.setLevel(logging.ERROR)

def log_amp_config(
    logger: logging.Logger, 
    accelerator: Accelerator, 
) -> None:
    """Log AMP-related configuration values in a consistent block."""
    if logger is None:
        return

    logger.info("Accelerator AMP Configuration:")
    logger.info(f"  mixed_precision: {accelerator.mixed_precision}")
    logger.info(f"  native_amp: {accelerator.native_amp}")


def log_allocated_gpu_memory(
    logger: logging.Logger,
    stage: str = "loading model",
    device: int = 0,
) -> None:
    """Log current GPU memory allocation if CUDA is available."""
    if logger is None:
        return

    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device)
        logger.info(
            f"Allocated GPU memory after {stage}: {allocated_memory/1024/1024/1024:.2f} GB"
        )
