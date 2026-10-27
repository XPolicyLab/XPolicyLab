"""Load WorldLA before native image libraries, then use the official WS server."""

import argparse
import asyncio
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    import yaml

    cfg = yaml.safe_load(args.config_path.read_text())
    for item in args.overrides:
        key, separator, value = item.partition("=")
        if not separator or key not in cfg:
            parser.error(f"Expected a declared deploy.yml key=value, got {item!r}")
        cfg[key] = yaml.safe_load(value)
    if cfg["protocol"] != "ws":
        parser.error("WorldLA supports the official ws protocol only")
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root.parent))
    sys.path.insert(0, str(root))
    from XPolicyLab.policy.WorldLA.model import Model

    model = Model(cfg)
    # setup_policy_server.py imports legacy TCP/image libraries at module load.
    # This adapter-specific launcher only changes import order, not the wire API.
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    server = PolicyServer(
        model, PolicyServerConfig(host=cfg["host"], port=int(cfg["port"]))
    )
    print(f"[WorldLA] official ws transport at {server.url}", flush=True)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
