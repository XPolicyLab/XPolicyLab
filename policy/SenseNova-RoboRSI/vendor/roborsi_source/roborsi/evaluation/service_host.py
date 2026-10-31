# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent HTTP service entry point for one RoboRSI Agent or Adapter episode."""

from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from roborsi.utils.config import get_project_env


def load_factory(specification: str) -> Callable[..., Any]:
    """Resolve a trusted ``module:function`` factory after process isolation."""
    module_name, separator, symbol = specification.partition(":")
    if not separator or not module_name or not symbol or "." in symbol:
        raise ValueError("factory must be a module:function reference")
    module = importlib.import_module(module_name)
    factory = getattr(module, symbol)
    if not callable(factory):
        raise TypeError(f"factory is not callable: {specification}")
    return factory


def main(argv: list[str] | None = None) -> None:
    """Run one case-scoped service until RPC shutdown or parent death."""
    parser = argparse.ArgumentParser(description="Host one RoboRSI episode service")
    parser.add_argument(
        "--kind", choices=("agent", "adapter", "benchmark"), required=True
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--factory", help="trusted module:function")
    source.add_argument("--agent-config", help="Agent YAML/JSON path or bundle:name")
    parser.add_argument(
        "--agent-set", action="append", default=[], help="PATH=JSON preset override"
    )
    parser.add_argument(
        "--config-sha256", help="expected Agent preset content identity"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument(
        "--source-root", type=Path, help="trusted Agent source snapshot"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.source_root is not None:
        if args.kind != "agent" or not args.source_root.is_dir():
            parser.error("--source-root requires an Agent source directory")
        sys.path.insert(0, str(args.source_root.resolve()))
    if args.config_sha256 and not args.agent_config:
        parser.error("--config-sha256 requires --agent-config")
    if args.agent_set and not args.agent_config:
        parser.error("--agent-set requires --agent-config")
    if args.agent_config:
        if args.kind != "agent":
            parser.error("--agent-config requires --kind agent")
        if args.source_root is not None:
            parser.error("Agent presets declare their own source_root")
        from roborsi.agents.config import AgentConfig

        config = AgentConfig.load(args.agent_config, overrides=args.agent_set)
        if args.config_sha256 and config.fingerprint != args.config_sha256:
            parser.error("Agent config inputs changed before service startup")
        if args.case_dir is None or not args.case_dir.is_dir():
            parser.error("--case-dir must be an existing directory for an agent")
        from roborsi.evaluation.services import AgentFacade

        facade = AgentFacade(config.create_agent(args.case_dir))
        facade.serve(
            transport="http",
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
        return
    factory = load_factory(args.factory)
    if args.kind == "benchmark":
        if args.case_dir is not None:
            parser.error("Benchmark output directories are private to the HTTP service")
        from roborsi.evaluation.benchmark_http import (
            BenchmarkApplication,
            BenchmarkHttpServer,
        )
        from roborsi.utils.daemon import watch_parent_death

        stopped = threading.Event()
        if args.parent_watch:
            watch_parent_death(stopped.set)
        with tempfile.TemporaryDirectory(prefix="roborsi-benchmark-") as root:
            app = BenchmarkApplication(factory(), Path(root))
            server = BenchmarkHttpServer(
                (args.host, args.port),
                app,
                token=get_project_env("ROBORSI_BENCHMARK_TOKEN"),
            )
            try:
                threading.Thread(target=server.serve_forever, daemon=True).start()
                stopped.wait()
            finally:
                app.close()
                server.shutdown()
                server.server_close()
        return
    if args.source_root is not None:
        module = sys.modules[factory.__module__]
        location = getattr(module, "__file__", None)
        if location is None or not Path(location).resolve().is_relative_to(
            args.source_root.resolve()
        ):
            parser.error("Agent factory was not loaded from --source-root")
    if args.kind == "agent":
        from roborsi.evaluation.services import AgentFacade

        if args.case_dir is None or not args.case_dir.is_dir():
            parser.error("--case-dir must be an existing directory for an agent")
        facade = AgentFacade(factory(args.case_dir.resolve()))
    else:
        if args.case_dir is not None:
            parser.error("--case-dir is only valid for an agent")
        from roborsi.evaluation.adapter_service import AdapterFacade

        facade = AdapterFacade(factory())
    facade.serve(
        transport="http",
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
