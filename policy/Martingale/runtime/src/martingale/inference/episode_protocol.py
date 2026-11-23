# Copyright © 2026 Martingale. All Rights Reserved.
"""Wire settings shared by the isolated policy runner and the Codex endpoint."""
import json

ENDPOINT = "wss://codex.martingale-robotics.com/episode"
MAX_MESSAGE = 32 * 1024 * 1024


def python_tool():
    return {
        "name": "python",
        "description": "Run a persistent Python cell. Use a new request_id per NEW cell; retry the same id only for a lost response. Variables persist. Never replay a partially completed cell. display(image) sends image bytes for the NEXT model decision.",
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"request_id": {"type": "string"}, "code": {"type": "string"}},
                         "required": ["request_id", "code"]},
    }


def parse_frame(raw):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected an episode message object")
    return value


def tool_response(result):
    items = []
    for block in result.content_blocks:
        if block["type"] == "text":
            items.append({"type": "inputText", "text": block["text"]})
        elif block["type"] == "image":
            source = block["source"]
            items.append({"type": "inputImage", "imageUrl":
                          "data:" + source["media_type"] + ";base64," + source["data"]})
        else:
            raise ValueError("Unsupported tool result content")
    return {"success": not bool(result.result.get("error")), "contentItems": items}
