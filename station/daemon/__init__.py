"""HTTP daemon for eval-station environment clients."""

from station.daemon.cli import (
    add_debug_env_client_arguments,
    baseline_from_args,
    main,
)
from station.daemon.handler import (
    create_server,
    make_handler,
    session_dispatch_path,
    session_start_path,
    session_stop_path,
)
from station.daemon.state import EnvClientServerConfig, EnvClientServerState
from station.dispatch.errors import normalize_execution_error

__all__ = [
    "EnvClientServerConfig",
    "EnvClientServerState",
    "add_debug_env_client_arguments",
    "baseline_from_args",
    "create_server",
    "main",
    "make_handler",
    "normalize_execution_error",
    "session_dispatch_path",
    "session_start_path",
    "session_stop_path",
]
