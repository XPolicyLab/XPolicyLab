#!/usr/bin/env python3
"""Serve one policy over a local JSON HTTP API (one episode at a time).

python scripts/serve.py --model robodojo --device cuda:0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prismwam.deployment.policy import load_policy
from prismwam.hub import bundle_files, resolve_bundle


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", help="model folder, or robotwin / robodojo / libero (downloaded)"
    )
    parser.add_argument("--config", help="instead of --model: config file")
    parser.add_argument("--checkpoint", help="instead of --model: weights file")
    parser.add_argument(
        "--statistics", help="action statistics; default: from the config"
    )
    parser.add_argument(
        "--evaluation", help="inference settings; default: the config's `deployment`"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.model:
        args.config, args.checkpoint = bundle_files(resolve_bundle(args.model))
    elif not (args.config and args.checkpoint):
        parser.error("Pass --model, or both --config and --checkpoint.")
    policy = load_policy(
        args.config,
        args.checkpoint,
        statistics_path=args.statistics,
        evaluation=args.evaluation,
        device=args.device,
    )

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            encoded = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path != "/healthz":
                return self.respond(404, {"error": "Unknown endpoint"})
            self.respond(
                200,
                {
                    "ready": True,
                    "model": "PrismWAM",
                    "action_dim": policy.action_dim,
                    "action_horizon": policy.action_horizon,
                    "replan_steps": policy.replan_steps,
                },
            )

        def do_POST(self):
            try:
                if self.path not in {"/infer", "/reset"}:
                    return self.respond(404, {"error": "Unknown endpoint"})
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 64 * 1024 * 1024:
                    return self.respond(
                        413, {"error": "Expected JSON body of at most 64 MiB"}
                    )
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("JSON request must be an object")
                if self.path == "/reset":
                    policy.reset()
                    return self.respond(200, {"reset": True})
                output = policy.predict(
                    request["observation"], seed=request.get("seed", 42)
                )
                output["action"] = output["action"].tolist()
                self.respond(200, output)
            except (ValueError, TypeError, KeyError) as error:
                self.respond(400, {"error": str(error)})
            except Exception:
                logging.exception("Policy inference failed")
                self.respond(
                    500, {"error": "Policy inference failed; inspect server logs"}
                )

    print(
        f"PrismWAM ready on http://{args.host}:{args.port}; one episode/session per process",
        flush=True,
    )
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
