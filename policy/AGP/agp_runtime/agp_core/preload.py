"""Take the task text and the first observation before the policy starts (preload switch).

The launcher calls the same `goal` and `observe` tools the policy would call, over the
episode's own MCP HTTP server, so the observation is created and journaled by the
unchanged runtime. The images are saved for `codex exec -i` and the compact state goes
into the launch message, which saves the policy its opening goal/observe turn.
"""
import asyncio
import base64
import json
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ORDER = ("top", "left_wrist", "right_wrist")


async def _fetch(url):
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as client:
            await client.initialize()
            goal = await client.call_tool("goal", {})
            observation = await client.call_tool("observe", {})
    return goal, observation


def preload(url, session: Path):
    goal, observation = asyncio.run(_fetch(url))
    instruction = json.loads(goal.content[0].text)["instruction"]
    state = observation.content[0].text
    if not json.loads(state).get("ok"):
        raise RuntimeError(f"first observation failed: {state}")
    folder = session / "preload"
    folder.mkdir(exist_ok=True)
    images, label = {}, None
    for block in observation.content[1:]:
        if block.type == "text" and block.text.startswith("Image "):
            label = block.text.split()[1]
        elif block.type == "image" and label in ORDER:
            path = folder / f"{label}.png"
            path.write_bytes(base64.b64decode(block.data))
            images[label] = path
    if set(images) != set(ORDER):
        raise RuntimeError(f"first observation images incomplete: {sorted(images)}")
    message = (f"\n# This episode\n\nTask: {instruction}\n\nThe first observation is already taken, so do not call "
               f"`goal` or `observe` to start. Its compact state:\n\n{state}\n\nThe three images attached to this "
               f"message are this observation's {', '.join(ORDER)} images, in that order.\n")
    (folder / "launch_message.md").write_text(message)
    return message, [images[name] for name in ORDER]
