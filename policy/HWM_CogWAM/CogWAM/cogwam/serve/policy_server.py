"""Launch the CogWAM policy server.

The wrapper owns un-normalization and chunk-size discovery, so the RoboDojo
client only forwards ``examples`` and consumes already-unnormalized actions
from the response.

Two checkpoint sources are supported:

``--artifact-dir``
    A released artifact produced by ``tools/convert_checkpoint.py``
    (``model.safetensors`` + ``dataset_statistics.json`` + ``artifact_manifest.json``
    + ``inference_config.yaml``). The manifest's sha256 entries are verified
    before anything is loaded.

``--ckpt_path``
    A raw training checkpoint inside ``<run>/checkpoints/``, resolved against
    the ``config.yaml`` / ``dataset_statistics.json`` saved beside it.

Backbone weights are NOT part of the artifact: the VLM and the DINO teacher are
resolved from ``COGWAM_BASE_VLM`` / ``COGWAM_DINO_MODEL``.
"""

from __future__ import annotations

import argparse
import logging
import socket

from cogwam.serve.policy_wrapper import PolicyServerWrapper
from cogwam.serve.protocol import WebsocketPolicyServer


def main(args) -> None:
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        artifact_dir=args.artifact_dir,
        device=args.device,
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
        dino_stats_path=args.dino_stats_path,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # Start the websocket server; wrapper.metadata is pushed at handshake.
    server = WebsocketPolicyServer(
        policy=wrapper,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    logging.info("server running ... metadata=%s", wrapper.metadata)
    server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-dir", type=str, default=None, help="released inference artifact directory")
    source.add_argument("--ckpt_path", type=str, default=None, help="training checkpoint (.pt or .safetensors)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--unnorm_key", type=str, default=None, help="dataset_statistics.json key; auto when unique")
    parser.add_argument(
        "--dino_stats_path",
        type=str,
        default=None,
        help="per-suite dino_v3_stats.json for online DINO normalization",
    )
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
