"""Offline CLI process that drives the real RoboRSI loopback MCP boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROGRAM = r"""
import json, os, pathlib, sys, urllib.request
if sys.argv[1:] == ["--version"]:
    print("codex-cli 0.162.1")
    raise SystemExit(0)
record = pathlib.Path(RECORD_PATH)
def save(value):
    with record.open("a") as stream:
        stream.write(json.dumps(value) + "\n")
def emit(value):
    print(json.dumps(value), flush=True)
save({"event":"started", "pid":os.getpid(), "argv":sys.argv,
      "prompt":sys.stdin.read(), "has_key":bool(os.environ.get("CODEX_API_KEY")),
      "trajectory_id":os.environ.get("ROBORSI_TRAJECTORY_ID")})
emit({"type":"thread.started", "thread_id":"offline-test"})
args = sys.argv
override = next(args[i+1] for i in range(len(args)-1)
                if args[i] == "-c" and args[i+1].startswith("mcp_servers.roborsi.url="))
url = json.loads(override.split("=", 1)[1])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def rpc(method, params, request_id):
    request = urllib.request.Request(
        url, json.dumps({"jsonrpc":"2.0", "id":request_id,
                         "method":method, "params":params}).encode(),
        headers={"Content-Type":"application/json",
                 "Accept":"application/json, text/event-stream"})
    with opener.open(request, timeout=15) as response:
        return json.load(response)["result"]
rpc("initialize", {"protocolVersion":"2024-11-05", "capabilities":{},
                   "clientInfo":{"name":"offline-model-test", "version":"1"}}, 1)
available = rpc("tools/list", {}, 10001)
names = {tool["name"] for tool in available["tools"]}
assert {"roborsi_action", "icl_list", "icl_read_text", "icl_read_image"} <= names
assert not {"robot.act", "vla_propose", "vla_review"} & names
save({"event":"tools", "names":sorted(names)})
listing = rpc("tools/call", {"name":"icl_list", "arguments":{"path":"."}}, 10002)
assert not listing.get("isError"), listing
save({"event":"icl_list", "result":listing})
entries = []
for block in listing.get("content", []):
    if block.get("type") == "text":
        entries.extend(json.loads(block["text"]).get("entries", []))
entry = next((item["path"] for item in entries if item["path"] == "START_HERE.md"),
             next(item["path"] for item in entries if item["type"] == "file" and item["path"].endswith(".md")))
reference = rpc("tools/call", {"name":"icl_read_text", "arguments":{"path":entry}}, 10003)
assert not reference.get("isError"), reference
save({"event":"icl_read", "path":entry, "result":reference})
for index in range(1000):
    arguments = json.loads(ACTION_JSON)
    save({"event":"call", "index":index, "arguments":arguments})
    result = rpc("tools/call", {"name":"roborsi_action", "arguments":arguments}, index+2)
    save({"event":"feedback", "index":index, "result":result})
    emit({"type":"item.completed", "item":{
        "id":str(index), "type":"mcp_tool_call", "server":"roborsi",
        "tool":"roborsi_action", "arguments":arguments,
        "result":result, "status":"completed"}})
    if result.get("isError"):
        emit({"type":"turn.failed", "error":{"message":"MCP action rejection"}})
        raise SystemExit(3)
    documents = []
    for block in result.get("content", []):
        if block.get("type") == "text":
            try:
                documents.append(json.loads(block["text"]))
            except (ValueError, TypeError):
                pass
    if any(document.get("_finish") for document in documents if isinstance(document, dict)):
        break
emit({"type":"item.completed", "item":{
    "id":"answer", "type":"agent_message", "text":"episode acknowledged"}})
emit({"type":"turn.completed", "usage":{
    "input_tokens":10, "cached_input_tokens":0, "output_tokens":4}})
"""


def write_cli(
    executable: Path,
    *,
    record_path: Path,
    action: dict,
    interpreter: str = sys.executable,
) -> Path:
    """Write an executable test CLI; callers pin its actual digest in fixtures."""
    executable.parent.mkdir(parents=True, exist_ok=True)
    program = _PROGRAM.replace("RECORD_PATH", repr(str(record_path))).replace(
        "ACTION_JSON",
        repr(json.dumps(action)),
    )
    executable.write_text(f"#!{interpreter}\n" + program)
    executable.chmod(0o755)
    return executable
