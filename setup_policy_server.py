import asyncio
import os
import threading
import ast
import time
import yaml
import importlib
import argparse
import traceback
from client_server.tcp.model_server import ModelServer


def _agent_debug_log(location, message, data, hypothesis_id="H0,H2,H4"):
    try:
        import json
        from pathlib import Path

        log_path = Path("/personal/tianxing/RoboDojo/XPolicyLab/.cursor/debug-0684e4.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": "0684e4",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def _default_protocol() -> str:
    """Default to the websocket policy protocol."""
    return "ws"

def eval_function_decorator(policy_model_name, Func_and_Class_name):
    """Load a specified function (e.g., get_model) from a policy module"""
    module = importlib.import_module(policy_model_name)
    return getattr(module, Func_and_Class_name)

def main(deploy_cfg):
    """Main entry: load model, start server, run indefinitely"""
    # Extract basic arguments
    policy_name = deploy_cfg.get("policy_name")
    port = deploy_cfg.get("port")
    host = deploy_cfg.get("host", "0.0.0.0")
    protocol = deploy_cfg.get("protocol", "ws")

    # region agent log
    _agent_debug_log(
        "setup_policy_server.py:main",
        "starting policy server",
        {
            "policy_name": policy_name,
            "host": host,
            "port": port,
            "protocol": protocol,
            "ckpt_name": deploy_cfg.get("ckpt_name"),
            "checkpoint_path": deploy_cfg.get("checkpoint_path"),
            "env_cfg_type": deploy_cfg.get("env_cfg_type"),
            "action_type": deploy_cfg.get("action_type"),
            "task_name": deploy_cfg.get("task_name"),
        },
    )
    # endregion agent log

    # Instantiate model
    model_class_func = eval_function_decorator(f"XPolicyLab.policy.{policy_name}.model", "Model")
    model = model_class_func(deploy_cfg)

    if protocol == "ws":
        try:
            from client_server.ws.model_server import PolicyServer, PolicyServerConfig
        except ModuleNotFoundError as exc:
            if exc.name == "client_server":
                # client_server.ws ships in this repo; make it importable even when
                # XPolicyLab is not pip-installed in the current environment.
                import sys
                repo_root = os.path.dirname(os.path.abspath(__file__))
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                try:
                    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
                except ModuleNotFoundError as dep_exc:
                    raise RuntimeError(
                        "ws policy server requires XPolicyLab websocket dependencies "
                        f"(missing module: {dep_exc.name}). Install in the policy env with: "
                        "pip install -e . from the XPolicyLab root."
                    ) from dep_exc
            else:
                raise RuntimeError(
                    "ws policy server requires XPolicyLab websocket dependencies "
                    f"(missing module: {exc.name}). Install in the policy env with: "
                    "pip install -e . from the XPolicyLab root."
                ) from exc

        server = PolicyServer(
            model,
            PolicyServerConfig(
                host=host,
                port=int(port),
            ),
        )
        try:
            asyncio.run(server.serve_forever())
        except KeyboardInterrupt:
            print("\nShutting down websocket policy server...")
        return
    if protocol != "legacy_tcp":
        raise ValueError(f"unsupported policy server protocol: {protocol}")

    # Wrap server.start so exceptions inside thread are fully printed
    def run_server():
        try:
            server.start()
        except Exception:
            print("\033[31m[ERROR] Exception occurred inside server thread:\033[0m")
            traceback.print_exc()
            raise

    # Start server in background thread
    server = ModelServer(model, host=host, port=port)
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Keep main thread alive until KeyboardInterrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.stop()
        thread.join()

def parse_args_and_config():
    """Parse CLI args and YAML config, merge overrides"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "--config-path", dest="config_path", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--protocol", choices=("legacy_tcp", "ws"), help="Policy server protocol")
    parser.add_argument("--host", help="Policy server bind host")
    parser.add_argument("--port", type=int, help="Policy server bind port")
    parser.add_argument("--relay-url", dest="relay_url", help="Relay URL for future relay mode")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, help="Override config values")
    args = parser.parse_args()

    # Load base config
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Parse overrides: --key value pairs

    def _parse_val(s: str):
        # safer than eval; supports numbers/bool/None/list/dict when properly quoted
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
        
    if args.overrides:
        tokens = args.overrides

        # Case A: key=value key=value ...
        if all(("=" in t and not t.startswith("-")) for t in tokens):
            for t in tokens:
                k, v = t.split("=", 1)
                cfg[k] = _parse_val(v)
        else:
            # Case B: --key value --key value ...
            if len(tokens) % 2 != 0:
                raise ValueError(f"--overrides expects key value pairs, got: {tokens}")

            it = iter(tokens)
            for key in it:
                val = next(it)
                cfg[key.lstrip("-")] = _parse_val(val)

    if args.protocol is not None:
        cfg["protocol"] = args.protocol
    else:
        cfg.setdefault("protocol", _default_protocol())
    if args.host is not None:
        cfg["host"] = args.host
    if args.port is not None:
        cfg["port"] = args.port
    if args.relay_url is not None:
        cfg["relay_url"] = args.relay_url

    # region agent log
    _agent_debug_log(
        "setup_policy_server.py:parse_args_and_config",
        "parsed policy server config",
        {
            "config_path": args.config_path,
            "overrides": args.overrides,
            "policy_name": cfg.get("policy_name"),
            "host": cfg.get("host"),
            "port": cfg.get("port"),
            "protocol": cfg.get("protocol"),
            "ckpt_name": cfg.get("ckpt_name"),
            "checkpoint_path": cfg.get("checkpoint_path"),
            "env_cfg_type": cfg.get("env_cfg_type"),
            "action_type": cfg.get("action_type"),
        },
    )
    # endregion agent log
    
    def _require_non_empty(key: str):
        if key not in cfg:
            raise ValueError(f"{key} must be specified in config or overrides")
        val = cfg[key]
        if val is None:
            raise ValueError(f"{key} must be non-empty")
        if isinstance(val, str) and not val.strip():
            raise ValueError(f"{key} must be non-empty")

    _require_non_empty("host")
    _require_non_empty("port")
    return cfg

if __name__ == "__main__":
    deploy_cfg = parse_args_and_config()
    main(deploy_cfg)
