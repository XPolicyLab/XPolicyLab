import asyncio
import os
import threading
import ast
import time
import yaml
import importlib
import argparse
import traceback
from client_server.model_server import ModelServer


def _default_protocol() -> str:
    """Pick the wire protocol matching the eval env client.

    The sim eval client (src/eval_client via ModelClient) only speaks the
    legacy length-prefixed TCP protocol, while the debug/real env clients
    speak robodojo_ws. EVAL_ENV_TYPE follows utils/resolve_eval_env_type.sh
    semantics: empty/unset means sim.
    """
    eval_env_type = (os.environ.get("EVAL_ENV_TYPE") or "sim").strip()
    return "legacy_tcp" if eval_env_type == "sim" else "robodojo_ws"

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
    protocol = deploy_cfg.get("protocol", "robodojo_ws")

    # Instantiate model
    model_class_func = eval_function_decorator(f"XPolicyLab.policy.{policy_name}.model", "Model")
    model = model_class_func(deploy_cfg)

    if protocol == "robodojo_ws":
        try:
            from eval_station.servers.policy_server import PolicyServer, PolicyServerConfig
        except ModuleNotFoundError as exc:
            if exc.name == "eval_station":
                # eval_station ships in this repo; make it importable even when
                # XPolicyLab is not pip-installed in the current environment.
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "integrations"))
                try:
                    from eval_station.servers.policy_server import PolicyServer, PolicyServerConfig
                except ModuleNotFoundError as dep_exc:
                    raise RuntimeError(
                        "robodojo_ws policy server requires the eval-station dependencies "
                        f"(missing module: {dep_exc.name}). Install them in the policy env with: "
                        "pip install 'websockets>=13' 'msgpack>=1.0.8' 'msgpack-numpy>=0.4.8' 'pydantic>=2.5' "
                        "(or pip install -e '.[eval-station]' from the XPolicyLab root)."
                    ) from dep_exc
            else:
                raise RuntimeError(
                    "robodojo_ws policy server requires the eval-station dependencies "
                    f"(missing module: {exc.name}). Install them in the policy env with: "
                    "pip install 'websockets>=13' 'msgpack>=1.0.8' 'msgpack-numpy>=0.4.8' 'pydantic>=2.5' "
                    "(or pip install -e '.[eval-station]' from the XPolicyLab root)."
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
            print("\nShutting down RoboDojo policy server...")
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
    parser.add_argument("--protocol", choices=("legacy_tcp", "robodojo_ws"), help="Policy server protocol")
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
    
    def _require_non_empty(key: str):
        if key not in cfg:
            raise ValueError(f"{key} must be specified in config or overrides")
        val = cfg[key]
        if val is None:
            raise ValueError(f"{key} must be non-empty")
        if isinstance(val, str) and not val.strip():
            raise ValueError(f"{key} must be non-empty")

    for deprecated_key in ("policy_server_host", "policy_server_port"):
        if deprecated_key in cfg:
            print(
                f"\033[31m[WARNING] Deprecated config key '{deprecated_key}' is present; "
                f"use 'host' and 'port' instead.\033[0m"
            )

    _require_non_empty("host")
    _require_non_empty("port")
    return cfg

if __name__ == "__main__":
    deploy_cfg = parse_args_and_config()
    main(deploy_cfg)
