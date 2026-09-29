#!/usr/bin/env python3
"""Launch RoboDojo with websocket settings safe for its synchronous client.

The upstream environment-side ``WsModelClient`` advances its private asyncio
loop only inside synchronous policy calls. Isaac reset and simulation work can
therefore pause that loop for more than the default 20-second websocket
keepalive window. Patch only the local simulator-to-XPolicy connection to
disable protocol pings, then execute the official RoboDojo evaluation entry.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import MethodType

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERLAY_ROOT = Path(__file__).resolve().parent / "xpolicy_overlay"


def _robodojo_root() -> Path:
    raw = os.environ.get("ROBODOJO_ROOT", "").strip()
    if not raw:
        raise ValueError("ROBODOJO_ROOT is not set; see README.md for the pinned benchmark checkout")
    return Path(raw).resolve()


def _prepend_import_paths(robodojo_root: Path) -> None:
    ordered = (OVERLAY_ROOT, REPO_ROOT, robodojo_root, robodojo_root / "XPolicyLab")
    resolved = {str(path) for path in ordered}
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) not in resolved]
    sys.path[:0] = [str(path) for path in ordered]


def _disable_sync_bridge_keepalive() -> None:
    module = importlib.import_module("client_server.ws.protocol.client")
    original = module.PolicyEvalClientConfig
    if getattr(original, "_cogwam_no_keepalive", False):
        return

    def policy_client_config_no_keepalive(*args, **kwargs):
        kwargs.setdefault("ws_ping_interval_s", None)
        kwargs.setdefault("ws_ping_timeout_s", None)
        return original(*args, **kwargs)

    policy_client_config_no_keepalive._cogwam_no_keepalive = True
    module.PolicyEvalClientConfig = policy_client_config_no_keepalive
    # Importing ``client_server.ws.protocol.client`` first executes the parent
    # ``client_server.ws`` package, whose __init__ eagerly imports
    # model_client. Replace that already-bound module global as well.
    model_client_module = sys.modules.get("client_server.ws.model_client")
    if model_client_module is not None:
        model_client_module.PolicyEvalClientConfig = policy_client_config_no_keepalive
    print("[RoboDojo] simulator websocket keepalive disabled", flush=True)


def _without_video(create_eval_env):
    """Wrap upstream env creation without changing observations or scoring."""

    def create_eval_env_without_video(*args, **kwargs):
        env = create_eval_env(*args, **kwargs)
        # No rollout has started yet, but clean defensively in case an upstream
        # constructor version created a writer while warming up cameras.
        abort = getattr(env, "_abort_video_writers", None)
        if callable(abort):
            abort()

        def ignore_vision_stream(self, env_idx, frame):
            _ = (self, env_idx, frame)

        def ignore_video_save(self, env_idx, video_path, tag):
            _ = (env_idx, video_path, tag)
            writers = getattr(self, "video_writers", {}).pop(env_idx, {})
            for writer in writers.values():
                try:
                    writer.abort()
                except Exception:
                    pass

        env._stream_vision = MethodType(ignore_vision_stream, env)
        env.save_video = MethodType(ignore_video_save, env)
        print(
            "[RoboDojo] fast rollout: video streaming/encoding disabled; "
            "policy observations and result JSON unchanged",
            flush=True,
        )
        return env

    return create_eval_env_without_video


def _install_eval_batch_override(namespace: dict) -> None:
    """Make the official client honor this invocation's runtime batch mode.

    Upstream ``main.py`` re-reads ``<ROBODOJO_ROOT>/XPolicyLab/policy/<name>/deploy.yml``
    after parsing ``--num_envs``. That shared checkout intentionally remains
    untouched and has no entry for this repository's policy name, which would
    silently collapse a requested six-environment rollout back to one. The
    repository launcher already owns the per-run deployment contract, so
    override only the lookup in this process.
    """

    raw = os.environ.get("ROBODOJO_EVAL_BATCH")
    if raw is None:
        return
    if not callable(namespace.get("_eval_batch_from_deploy")):
        raise RuntimeError(
            "RoboDojo official main no longer exposes _eval_batch_from_deploy; "
            "refusing to silently reduce vector rollout concurrency."
        )
    normalized = raw.strip().lower()
    if normalized not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise ValueError(f"ROBODOJO_EVAL_BATCH must be a boolean, got {raw!r}")
    enabled = normalized in {"1", "true", "yes", "on"}
    namespace["_eval_batch_from_deploy"] = lambda _policy_name: enabled
    print(
        f"[RoboDojo] simulator batch contract overridden by launcher: eval_batch={str(enabled).lower()}",
        flush=True,
    )


def _execute_official_main(entry: Path) -> None:
    """Define upstream main, apply narrow runtime hooks, then call it.

    Executing with a non-``__main__`` name prevents the file's trailing main
    call from firing before hooks are installed. Keeping ``sys.argv[0]`` as
    this launcher also ensures RoboDojo's os.execv PhysX recovery re-enters the
    launcher and retains both the keepalive and no-video settings.
    """

    namespace = {
        "__name__": "_cogwam_robodojo_official_main",
        "__file__": str(entry),
        "__package__": None,
        "__cached__": None,
    }
    source = entry.read_bytes()
    exec(compile(source, str(entry), "exec"), namespace)
    _install_eval_batch_override(namespace)
    if os.environ.get("ROBODOJO_DISABLE_EVAL_VIDEO", "0") == "1":
        namespace["create_eval_env"] = _without_video(namespace["create_eval_env"])
    namespace["main"]()


def main() -> None:
    robodojo_root = _robodojo_root()
    _prepend_import_paths(robodojo_root)
    _disable_sync_bridge_keepalive()
    entry = robodojo_root / "src/eval_client/main.py"
    if not entry.is_file():
        raise FileNotFoundError(f"missing RoboDojo evaluation entry: {entry}")
    _execute_official_main(entry)


if __name__ == "__main__":
    main()
