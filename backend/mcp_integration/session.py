"""
MCPSession — async context manager that binds MCP servers to an agent.

The single entry point for connecting MCP, converting their tools,
registering them on an agent, and cleaning up — all within one
``async with`` block.

Usage
-----
    agent = MonoAgent(provider=...)
    async with MCPSession(agent) as session:
        await session.add_server("fs", {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        })
        response = await agent.run("Read /tmp/notes.txt")
    # ── all MCP connections closed automatically ──

What happens on ``add_server()``:
  1. Connects to the MCP server (stdio / SSE)
  2. Lists all tools exposed by the server
  3. Converts them to ``MCPTool`` instances via ``MCPToolConverter``
  4. Registers every tool in the agent's ``ToolRegistry``

On context exit (``__aexit__``):
  - Every connected server is cleanly shut down
  - Transport streams closed, subprocesses terminated
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from backend.mcp_integration.manager import MCPManager
from backend.mcp_integration.converter import MCPToolConverter
from backend.tool_registry import ToolRegistry

if TYPE_CHECKING:
    from backend.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MCPSession:
    """
    Async context manager that connects MCP servers and binds their
    tools onto an agent.

    Parameters
    ----------
    agent:
        The agent instance that will receive the MCP tools.  The
        session registers tools on ``agent.tools`` (a ``ToolRegistry``),
        creating one if the agent doesn't already have one.
    converter:
        Optional custom ``MCPToolConverter``.  Defaults to a standard
        instance that prefixes tool names with ``{server_name}_``.
    """

    def __init__(
        self,
        converter: MCPToolConverter | None = None,
    ) -> None:
        self._converter = converter or MCPToolConverter()
        self._manager = MCPManager(converter=self._converter)
        self._registry: ToolRegistry | None = None

    # ── Context manager ──────────────────────────────────────────────

    async def __aenter__(self) -> MCPSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self._manager.shutdown()

    # ── Server management ────────────────────────────────────────────

    async def add_server(self, name: str, transport_config: dict) -> ToolRegistry:
        """
        Connect to an MCP server, convert its tools, and register them
        on the agent's ``ToolRegistry``.

        Parameters
        ----------
        name:
            Unique server label — also used as a prefix for every tool
            name (e.g. ``"filesystem"`` produces tools named
            ``filesystem_read_file``, ``filesystem_write_file``, etc.)
        transport_config:
            Connection configuration — same format as ``MCPClient``::

                {"type": "stdio", "command": "npx", "args": [...]}
                {"type": "sse", "url": "http://localhost:8000/sse"}
        """
        client = await self._manager.add_server(name, transport_config)
        mcp_tools = await client.list_tools()
        registry = self._ensure_registry()
        self._converter.convert_and_register(name, mcp_tools, client, registry)
        logger.info(
            "[MCPSession] Server '%s' registered %d tools",
            name,
            len(mcp_tools),
        )

        return registry

    async def remove_server(self, name: str) -> None:
        """
        Disconnect a server and remove its tools from the agent's
        ``ToolRegistry``.
        """
        client = self._manager.get_client(name)
        if client is None:
            return
        if self._registry:
            # Remove all tools matching the server prefix
            prefix = f"{name}_"
            to_remove = [
                tname
                for tname in list(self._registry._registry.keys())
                if tname.startswith(prefix)
            ]
            for tname in to_remove:
                self._registry._registry.pop(tname, None)
        await self._manager.remove_server(name)

    # ── Internals ────────────────────────────────────────────────────

    def _ensure_registry(self) -> ToolRegistry:
        """Return the agent's ``ToolRegistry``, creating one if needed."""
        if self._registry is not None:
            return self._registry
        self._registry = ToolRegistry()
        return self._registry
