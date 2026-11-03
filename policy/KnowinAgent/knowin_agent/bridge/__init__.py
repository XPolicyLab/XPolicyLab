"""Run the KnowinAgent agents inside an external coding-agent CLI (e.g. Codex) through MCP.

host.py owns the robot, the runtime and the episode state and serves tool calls
over a Unix socket; mcp.py is a stdio MCP server that forwards one role's tools
to the host. Every tool result carries a fresh context snapshot (status text and
the current camera images), which plays the role of h_{t+1}.
"""
