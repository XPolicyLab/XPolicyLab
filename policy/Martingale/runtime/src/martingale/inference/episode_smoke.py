"""Verify the public evaluator path with a tool result known only to the caller."""
import argparse
import base64
from pathlib import Path
import secrets
import struct
from types import SimpleNamespace
import zlib

from .config import InferenceConfig
from .episode_client import RemoteEpisode
from .episode_protocol import ENDPOINT


def probe_image():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rgb = b"\xff\x00\x00"
    raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", 64, 64, 8, 2, 0, 0, 0))
    raw += chunk(b"IDAT", zlib.compress((b"\x00" + rgb * 64) * 64)) + chunk(b"IEND", b"")
    return base64.b64encode(raw).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", default=ENDPOINT)
    args = parser.parse_args()
    config = InferenceConfig.load(args.config)
    run = RemoteEpisode(args.token_file.read_text().strip(), config.model, config.effort, args.url)
    nonce = secrets.token_hex(8)
    events = []
    calls = []

    def execute(name, arguments):
        if name != "python" or not arguments.get("request_id"):
            raise RuntimeError("Missing Python cell identity")
        calls.append(arguments["request_id"])
        return SimpleNamespace(result={"stdout": nonce}, content_blocks=[
            {"type": "text", "text": nonce},
            {"type": "image", "source": {"media_type": "image/png", "data": probe_image()}}])

    try:
        run.turn("Call the python tool with a new request_id and code print('probe'). "
                 "It returns a verification word and an image. Reply with the word and the image color.",
                 SimpleNamespace(execute_tool=execute), events.append, lambda: False)
        reply = " ".join(e["item"]["text"] for e in events if e["type"] == "item.completed")
        if not calls or nonce not in reply or "red" not in reply.lower():
            raise RuntimeError("Codex did not confirm both the caller's text and image")
        print("PASS: authenticated episode, live Codex turn, Python callback, text and camera-image response")
    finally:
        run.release()


if __name__ == "__main__":
    main()
