"""Optional MCP adapter; execution remains in the shared capability gateway.

Tool list/call registration follows the upstream bridge, without planner or
provider-shaped results. The stdio lifecycle is owned by the connecting client.
"""

import asyncio
import json


def create_server(gateway):
    from mcp import types
    from mcp.server import Server

    server = Server("physicalrsi")

    @server.list_tools()
    async def list_tools():
        return [types.Tool(**spec) for spec in gateway.specifications()]

    @server.call_tool()
    async def call_tool(name, arguments):
        reply = await asyncio.to_thread(gateway.call, name, arguments or {})
        content = [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"error": reply.error} if reply.error else reply.data,
                    ensure_ascii=False,
                    default=lambda value: value.tolist(),
                ),
            )
        ]
        for media in reply.media:
            content.append(
                types.ImageContent(
                    type="image", data=media["base64"], mimeType=media["mime_type"]
                )
            )
        return types.CallToolResult(content=content, isError=reply.error is not None)

    return server


async def serve(gateway):
    from mcp.server.stdio import stdio_server

    server = create_server(gateway)
    async with stdio_server() as (incoming, outgoing):
        await server.run(incoming, outgoing, server.create_initialization_options())


def main(argv=None):
    import argparse
    from pathlib import Path

    from PhysicalRSI.application import Application

    parser = argparse.ArgumentParser(description="physicalRSI MCP stdio server")
    parser.add_argument("--workspace", type=Path, default=Path(".physicalrsi"))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)
    application = Application(args.workspace)
    if args.demo:
        application.demo()
    asyncio.run(serve(application.tools()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
