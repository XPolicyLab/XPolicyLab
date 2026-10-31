"""OEH-condition client: the pinned LIBERO-family single_eval.py with OEH native receipts.

    python -m oeh_policy.libero_eval [--oeh-config <frozen oeh.policy.libero.v1>] -- <single_eval.py arguments>

The official `main`/`run_eval` still own the CLI merge with `policy_config.yml`,
seed, reset, init state, 30 settling steps, termination, native success and
`results.json` (`oeh_eval.libero_official`, shared with the direct condition). This
module substitutes only the official client's transport (ticketed, one-shot, OEH
events forwarded to the sidecar) and a native-step gate around its env.
`native-submissions.json` is kept next to the sidecar as an extra attachment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .libero_boundary import BENCHMARKS
from .libero_client import NativeEnvGateway, NativePolicyClient, ReceiptTransport
from .libero_config import check_client, resolve_config_path


def loop_sources(benchmark: str) -> list[str]:
    folder = BENCHMARKS[benchmark]
    return [f"benchmarks/{folder}/single_eval.py", f"benchmarks/{folder}/openwam2libero_interface.py",
            "benchmarks/utils/client.py", "benchmarks/utils/transport.py",
            "benchmarks/utils/action_conversion.py"]


def transport_timeout(cfg: dict, *, first: bool = False) -> float:
    """How long one released-client request may wait on the OEH server, with a margin.

    A fast step is one decision. A hybrid step can be a refused fast decision, the end of
    its brain turn, a full slow decision and, when that fails, the fast decision that
    replaces it, each bounded by `decision_seconds` (as `oeh_policy.model` bounds RoboDojo).
    The episode's first request may also spend what `first_decision_seconds` allows beyond
    `decision_seconds` on the model's warm-up. A review step ends within `review.call_seconds`
    (OpenWAM's candidate, the wait for the teacher and the hold that replaces a late decision),
    the first one within `first_call_extra_seconds` more, as `oeh_policy.model` bounds RoboDojo's.
    """
    if cfg["mode"] == "review":
        review = cfg["review"]
        return float(review["call_seconds"]) + (float(review["first_call_extra_seconds"]) if first else 0.0) + 15.0
    seconds = float(cfg["decision_seconds"])
    warmup = max(0.0, float(cfg.get("first_decision_seconds", seconds)) - seconds) if first else 0.0
    return (4 if cfg["mode"] == "hybrid" else 1) * seconds + warmup + 15.0


def check_request_timeout(cfg: dict, kwargs: dict) -> None:
    """The review arm gets no more time per call than the official client gives the direct arm: its first call's
    bound (``transport_timeout``) must fit the client's ``request_timeout`` (single_eval.py passes policy_config.yml's
    value, 300 s at fa431ec; 300 is also its default). ``oeh_eval.matrix_emit`` checks the same bound when it
    compiles the condition; this repeats it against the merged config the official loop actually built."""
    allowed = float(kwargs.get("request_timeout", 300))
    longest = transport_timeout(cfg, first=True)
    if cfg["mode"] == "review" and longest > allowed:
        raise RuntimeError(f"a review call may take {longest:g} s, more than the official client's request_timeout "
                           f"{allowed:g} s")


def main(argv=None):
    from oeh_eval.libero_evidence import WireEvidence
    from oeh_eval.libero_official import run, split_argv

    own, official = split_argv(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oeh-config", type=str)
    args = parser.parse_args(own)
    cfg = json.loads(resolve_config_path(args.oeh_config).read_text(encoding="utf-8"))
    benchmark = cfg.get("benchmark")
    if benchmark not in BENCHMARKS:
        raise ValueError("LIBERO, LIBERO-Plus or LIBERO-Pro required")
    if cfg.get("mode") not in ("fast", "hybrid", "review"):
        raise ValueError("LIBERO runs the fast, hybrid and review conditions only")
    check_client(cfg, loop_sources(benchmark))
    events = os.environ.get("OEH_EVAL_EVENTS")
    if not events:
        raise RuntimeError("OEH_EVAL_EVENTS must name the attempt's fresh evidence sidecar")
    evidence = WireEvidence(Path(events))
    native = {"env": None, "client": None}

    def wrap_env(env):
        if native["env"] is not None:
            raise RuntimeError("official one-episode lifecycle changed")
        native["env"] = NativeEnvGateway(env, None, evidence)
        return native["env"]

    def make_client(kwargs, interface):
        if native["env"] is None or native["client"] is not None:
            raise RuntimeError("official one-episode lifecycle changed")
        url = f"ws://{kwargs.get('host', '127.0.0.1')}:{kwargs.get('port', 8848)}"
        check_request_timeout(cfg, kwargs)
        transport = ReceiptTransport(interface.WSPolicyClient(url, timeout=transport_timeout(cfg, first=True)),
                                     evidence, steady_timeout=transport_timeout(cfg))
        client = NativePolicyClient(transport, interface.native_eef10_to_libero7d,
                                    interface.libero_obs_to_eef10, evidence)
        native["env"].client = native["client"] = client
        return client

    try:
        # The returned integer is the official runner's process status. The native
        # outcome is its untouched results.json; no OEH success predicate appears.
        return run(Path(cfg["source_root"]), benchmark, official, make_client=make_client, wrap_env=wrap_env)
    finally:
        client = native["client"]
        (Path(events).parent / "native-submissions.json").write_text(json.dumps(
            client.native_submissions if client is not None else [], indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
