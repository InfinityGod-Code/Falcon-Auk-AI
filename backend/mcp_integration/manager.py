"""
MCPManager — orchestrates connections to multiple MCP servers.

Manages the lifecycle of several ``MCPClient`` instances and aggregates
the tools they expose into a flat list of ``MCPTool`` adapters that can
be registered in ``ToolRegistry``.

Usage
-----
    manager = MCPManager()

    manager.add_server("filesystem", {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    })

    for tool in manager.get_all_tools():
        tool_registry.register_tool(tool)

    # On shutdown:
    manager.shutdown()
"""

from __future__ import annotations

import logging
from typing import Any
from backend.mcp_integration.mcp_client import MCPClient
from backend.mcp_integration.mcp_tool import MCPTool

logger = logging.getLogger(__name__)


class MCPManager:
    """
    Orchestrator for zero or more MCP server connections.

    Each server is identified by a unique ``name``.  Tools are exposed
    with the convention ``{server_name}_{tool_name}`` to avoid name
    collisions across servers.
    """

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}

    # ── Lifecycle ────────────────────────────────────────────────────

    def add_server(self, name: str, transport_config: dict) -> MCPClient:
        """
        Connect to an MCP server and register it.

        Parameters
        ----------
        name:
            Unique identifier for this server (used as a prefix for its
            tools to avoid collisions).
        transport_config:
            Passed verbatim to ``MCPClient.__init__``.  Examples::

                # stdio (local subprocess)
                {"type": "stdio", "command": "npx",
                 "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}

                # SSE (remote HTTP stream)
                {"type": "sse", "url": "http://localhost:8000/sse"}

        Raises
        ------
        ValueError
            If a server with the same *name* is already registered.
        MCPConnectionError
            If the connection could not be established.
        """
        if name in self._clients:
            raise ValueError(
                f"MCP server '{name}' is already connected. "
                f"Call remove_server('{name}') first if you intend to reconnect."
            )

        client = MCPClient(server_name=name, transport_config=transport_config)
        client.connect()
        self._clients[name] = client
        logger.info(
            "[MCPManager] Added server '%s' (%s transport)",
            name,
            transport_config.get("type"),
        )
        return client

    def remove_server(self, name: str) -> None:
        """
        Disconnect from an MCP server and unregister it.

        All tools previously exported by this server will stop working.
        Safe to call on a name that was never registered (no-op).
        """
        client = self._clients.pop(name, None)
        if client:
            client.disconnect()
            logger.info("[MCPManager] Removed server '%s'", name)

    def shutdown(self) -> None:
        """
        Disconnect from every registered MCP server.

        Call this during application teardown (e.g. ``finally`` block or
        signal handler) to ensure clean closure of all transport streams
        and event-loop threads.
        """
        for name in list(self._clients):
            self.remove_server(name)
        logger.info("[MCPManager] All servers shut down")

    # ── Queries ──────────────────────────────────────────────────────

    def get_client(self, name: str) -> MCPClient | None:
        """Return the ``MCPClient`` for a given server, or ``None``."""
        return self._clients.get(name)

    @property
    def server_names(self) -> list[str]:
        """Return the names of all connected servers."""
        return list(self._clients.keys())

    def get_all_tools(self) -> list[MCPTool]:
        """
        Aggregate tools from every connected server.

        Tool names are prefixed with ``{server_name}_`` to guarantee
        uniqueness when registered in a single ``ToolRegistry``.

        Returns
        -------
        list[MCPTool]
            Empty list if no servers are connected.
        """
        tools: list[MCPTool] = []
        for server_name, client in self._clients.items():
            if not client.is_connected:
                logger.warning(
                    "[MCPManager] Server '%s' is not connected, skipping", server_name
                )
                continue
            try:
                mcp_tools = client.list_tools()
            except Exception:
                logger.exception(
                    "[MCPManager] Failed to list tools from '%s'", server_name
                )
                continue
            for mt in mcp_tools:
                prefixed_name = f"{server_name}_{mt.name}"
                tools.append(
                    MCPTool(
                        name=prefixed_name,
                        description=mt.description or "",
                        input_schema=mt.inputSchema,
                        mcp_client=client,
                        server_name=server_name,
                        original_tool_name=mt.name,
                    )
                )
        return tools

    def get_all_tool_dicts(self) -> list[dict[str, Any]]:
        """
        Convenience: return every tool as a dict with keys
        ``server_name``, ``tool_name``, ``description``, ``schema``.
        Useful for introspection / dashboard displays.
        """
        entries: list[dict[str, Any]] = []
        for server_name, client in self._clients.items():
            if not client.is_connected:
                continue
            try:
                for mt in client.list_tools():
                    entries.append(
                        {
                            "server_name": server_name,
                            "tool_name": mt.name,
                            "qualified_name": f"{server_name}_{mt.name}",
                            "description": mt.description or "",
                            "schema": mt.inputSchema,
                        }
                    )
            except Exception:
                logger.exception(
                    "[MCPManager] Failed to list tools from '%s'", server_name
                )
        return entries
