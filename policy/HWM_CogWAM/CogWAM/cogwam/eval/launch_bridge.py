#!/usr/bin/env python3
"""Start RoboDojo's XPolicy bridge with a repository-owned import contract.

The bridge is a thin local WebSocket server the simulator talks to; it hosts
:class:`cogwam.eval.robodojo_policy.Model`, which in turn talks to the remote
CogWAM policy server. Running it through this launcher (instead of RoboDojo's
own ``setup_policy_server.py``) makes the overlay import deterministic and
supplies the narrow websockets compatibility the simulator image needs.

RoboDojo's client image contains websockets 12, while the checked-out
XPolicyLab websocket server imports the ``websockets.asyncio`` API introduced
in websockets 13. The legacy server API available in version 12 has the same
runtime operations used by XPolicyLab, so expose only the three required names
when the newer namespace isn't available. No package installation or edits to
the shared RoboDojo checkout are required.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERLAY_ROOT = Path(__file__).resolve().parent / "xpolicy_overlay"
POLICY_NAME = "cogwam"


def _robodojo_root() -> Path:
    raw = os.environ.get("ROBODOJO_ROOT", "").strip()
    if not raw:
        raise ValueError("ROBODOJO_ROOT is not set; see README.md for the pinned benchmark checkout")
    return Path(raw).resolve()


def _prepend_import_paths(robodojo_root: Path) -> None:
    # Remove duplicates first so these paths really are the first four entries.
    ordered = (OVERLAY_ROOT, REPO_ROOT, robodojo_root, robodojo_root / "XPolicyLab")
    resolved = {str(path) for path in ordered}
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) not in resolved]
    sys.path[:0] = [str(path) for path in ordered]


def _verify_overlay() -> None:
    module = importlib.import_module(f"XPolicyLab.policy.{POLICY_NAME}.model")
    actual = Path(module.__file__).resolve()
    expected = (OVERLAY_ROOT / f"XPolicyLab/policy/{POLICY_NAME}/model.py").resolve()
    if actual != expected:
        raise RuntimeError(f"RoboDojo loaded the wrong CogWAM adapter: expected {expected}, got {actual}")
    print(f"[RoboDojo] verified CogWAM adapter={actual}", flush=True)


def _install_websockets_12_server_compat() -> None:
    try:
        importlib.import_module("websockets.asyncio.server")
        import websockets

        print(f"[RoboDojo] websockets={websockets.__version__}; using native asyncio server", flush=True)
        return
    except ModuleNotFoundError as exc:
        if exc.name not in {"websockets.asyncio", "websockets.asyncio.server"}:
            raise

    import websockets

    try:
        from websockets.legacy.server import WebSocketServer, WebSocketServerProtocol, serve
    except ImportError as exc:
        raise RuntimeError(
            "RoboDojo requires either websockets>=13 or the legacy server API provided by websockets 12; "
            f"found websockets={websockets.__version__}"
        ) from exc

    asyncio_module = types.ModuleType("websockets.asyncio")
    asyncio_module.__path__ = []  # type: ignore[attr-defined]
    server_module = types.ModuleType("websockets.asyncio.server")
    server_module.Server = WebSocketServer
    server_module.ServerConnection = WebSocketServerProtocol
    server_module.serve = serve

    asyncio_module.server = server_module  # type: ignore[attr-defined]
    websockets.asyncio = asyncio_module  # type: ignore[attr-defined]
    sys.modules["websockets.asyncio"] = asyncio_module
    sys.modules["websockets.asyncio.server"] = server_module
    print(
        f"[RoboDojo] websockets={websockets.__version__}; enabled XPolicy legacy-server compatibility",
        flush=True,
    )


def _disable_sync_bridge_keepalive() -> None:
    """Disable protocol pings on the simulator-facing local bridge.

    RoboDojo's ``WsModelClient`` owns an asyncio loop that only runs while a
    synchronous model call is active. During Isaac reset/physics work that loop
    can legitimately be dormant for longer than XPolicyLab's 20-second
    keepalive window, so server pings produce a false 1011 timeout. RPC request
    timeouts and reconnect handling remain enabled at the protocol layer; only
    websocket control-frame keepalive is disabled.
    """

    module = importlib.import_module("client_server.ws.model_server")
    original = module.PolicyServerConfig
    if getattr(original, "_cogwam_no_keepalive", False):
        return

    def policy_server_config_no_keepalive(*args, **kwargs):
        kwargs.setdefault("ws_ping_interval_s", None)
        kwargs.setdefault("ws_ping_timeout_s", None)
        return original(*args, **kwargs)

    policy_server_config_no_keepalive._cogwam_no_keepalive = True
    module.PolicyServerConfig = policy_server_config_no_keepalive
    print("[RoboDojo] local XPolicy server websocket keepalive disabled", flush=True)


def _load_xpolicy_setup_module(robodojo_root: Path):
    setup_path = robodojo_root / "XPolicyLab/setup_policy_server.py"
    spec = importlib.util.spec_from_file_location("_robodojo_xpolicy_setup", setup_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load XPolicy setup module: {setup_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    robodojo_root = _robodojo_root()
    _prepend_import_paths(robodojo_root)
    _verify_overlay()
    _install_websockets_12_server_compat()
    _disable_sync_bridge_keepalive()
    setup = _load_xpolicy_setup_module(robodojo_root)
    setup.main(setup.parse_args_and_config())


if __name__ == "__main__":
    main()
