"""Tools of MCP servers (optional): the services of kind mcp an agent names in
[services] mcp = [...]. mcp_tools() lists every tool of those servers, mcp_call(tool,
**arguments) calls one. Adding a server is a service manifest and one line of the agent."""

from harness_policy.services.mcp import McpClient, McpError


class McpTools:
    def _mcp_index(self):
        """{tool name: (server, the server's name for it)}; a name two servers share is
        listed as "server.tool" for each. A server's tools are listed once; a server that
        does not answer is left out (its error in _mcp_errors) and asked again next time,
        so the other servers' tools stay usable."""
        clients = self.__dict__.setdefault("_mcp_clients", {})
        listed = self.__dict__.setdefault("_mcp_server_tools", {})
        errors = {}
        for server, config in (getattr(self, "mcp_servers", None) or {}).items():
            if server in listed:
                continue
            client = clients.get(server) or clients.setdefault(server, McpClient(
                config["url"], timeout=float(config.get("timeout_seconds", 60.0)),
                on_demand=config.get("on_demand")))
            try:
                listed[server] = client.tools()
            except Exception as error:  # reported with the tools, not raised for every server
                errors[server] = f"{type(error).__name__}: {error}"
        self.__dict__["_mcp_errors"] = errors
        seen = {}
        for server, tools in listed.items():
            for tool in tools:
                seen.setdefault(tool["name"], []).append((server, tool))
        index = {}
        for name, entries in seen.items():
            for server, tool in entries:
                index[name if len(entries) == 1 else f"{server}.{name}"] = (server, tool)
        return index

    def mcp_tools(self):
        """Every tool of the agent's MCP servers: [{"name", "server", "description",
        "input_schema"}], by the name mcp_call takes. The tools of a server that does not
        answer are missing; when no server answers, McpError says why."""
        index = self._mcp_index()
        errors = self.__dict__["_mcp_errors"]
        if not index and errors:
            raise McpError(f"No MCP server answered: {errors}")
        return [{"name": name, "server": server, "description": tool.get("description", ""),
                 "input_schema": tool.get("inputSchema", {})}
                for name, (server, tool) in sorted(index.items())]

    def mcp_call(self, tool, /, **arguments):
        """Call `tool` (a name from mcp_tools()) with keyword arguments as its input;
        returns its structured result, or its text. A tool error raises McpError."""
        index = self._mcp_index()
        if tool not in index:
            errors = self.__dict__["_mcp_errors"]
            raise ValueError(f"No MCP tool {tool!r}; tools: {sorted(index)}"
                             + (f"; servers that did not answer: {errors}" if errors else ""))
        server, spec = index[tool]
        return self.__dict__["_mcp_clients"][server].call(spec["name"], arguments)
