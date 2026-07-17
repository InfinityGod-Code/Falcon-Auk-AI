"""
MCPToolConverter — abstracts the conversion of MCP ``types.Tool``
definitions into the framework's ``MCPTool`` adapter instances.

This makes the conversion strategy pluggable.  Subclass to add custom
naming conventions, schema transformations, or filtering logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.mcp_integration.mcp_tool import MCPTool
from backend.tool_registry import ToolRegistry

if TYPE_CHECKING:
    from mcp import types as mcp_types
    from backend.mcp_integration.mcp_client import MCPClient


class MCPToolConverter:
    """
    Converts MCP tool definitions (``mcp.types.Tool``) into
    ``MCPTool`` instances that the framework can register in a
    ``ToolRegistry`` and execute via ``ToolRunner``.

    The default implementation prefixes each tool name with
    ``{server_name}_`` to guarantee uniqueness across servers
    and preserves the original MCP name in ``original_tool_name``
    so the ``MCPClient`` can call the right server-side tool.
    """

    def convert(
        self,
        server_name: str,
        mcp_tools: list[mcp_types.Tool],
        client: MCPClient,
    ) -> list[MCPTool]:
        """
        Convert a list of MCP tool definitions into framework MCPTool instances.

        Parameters
        ----------
        server_name:
            Used as a prefix for the tool's registry name.
        mcp_tools:
            Raw tool definitions from ``MCPClient.list_tools()``.
        client:
            The client that owns the connection to the server
            exposing these tools.

        Returns
        -------
        list[MCPTool]
        """
        result: list[MCPTool] = []
        for mt in mcp_tools:
            prefixed = f"{server_name}_{mt.name}"
            result.append(
                MCPTool(
                    name=prefixed,
                    description=mt.description or "",
                    input_schema=mt.inputSchema,
                    mcp_client=client,
                    server_name=server_name,
                    original_tool_name=mt.name,
                )
            )
        return result

    def convert_and_register(
        self,
        server_name: str,
        mcp_tools: list[mcp_types.Tool],
        client: MCPClient,
        registry: ToolRegistry,
    ) -> list[MCPTool]:
        """
        Convert and immediately register every tool in *registry*.

        Returns the list of created ``MCPTool`` instances so callers
        can inspect or further customise them.
        """
        tools = self.convert(server_name, mcp_tools, client)
        for tool in tools:
            registry.register_tool(tool)
        return tools
