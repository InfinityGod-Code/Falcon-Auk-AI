"""
MCPTool — adapter that wraps an MCP tool into our framework's ``Tool`` ABC.

An ``MCPTool`` is registered in ``ToolRegistry`` alongside regular
``FalconAukTool`` instances.  The agent loop, ``ToolRunner``, and provider
adapters all treat it identically — the MCP transport is hidden behind
the ``Tool`` interface.

Integration flow
----------------
    LLM responds with tool_call(name="filesystem_read", args={path: ...})
        │
        ▼
    ToolRunner.execute(tc)
        │
        ├── registry.get_tool("filesystem_read") → MCPTool
        │
        ├── tool.func(**args)          ← sync (FalconAukTool compat)
        └── await tool.async_run(...)  ← async (preferred from async pipeline)
              │
              └── await MCPClient.call_tool(name, kwargs)
                    │
                    └── ClientSession → MCP Server
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.core.base.models.model import ModelProvider
from backend.core.base.tools.tool import Tool

if TYPE_CHECKING:
    from backend.mcp_integration.mcp_client import MCPClient


class MCPTool(Tool):
    """
    Wraps a remote MCP tool so it appears as a local ``Tool``.

    Parameters
    ----------
    name:
        Prefixed name used when registering in ToolRegistry (e.g.
        ``"filesystem_read"``).  The LLM sees this name.
    description:
        Human-readable description forwarded to the LLM.
    input_schema:
        JSON Schema dictionary describing the expected arguments.
    mcp_client:
        The ``MCPClient`` instance that owns the connection to the server
        exposing this tool.
    server_name:
        Label used in error messages and logging.
    original_tool_name:
        The tool's name as the MCP server knows it (e.g. ``"read"``).
        Defaults to *name* when the tool doesn't need a prefix.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        mcp_client: MCPClient,
        server_name: str,
        original_tool_name: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._client = mcp_client
        self._server_name = server_name
        self._mcp_name = original_tool_name or name
        self.func = self.run

    def run(self, **kwargs: Any) -> str:
        """
        Synchronous execution — used by ``ToolRunner.execute()`` when the
        caller is not in an async context.  Falls back to the async path
        via ``asyncio.run()``.
        """
        import asyncio

        return asyncio.run(self.async_run(**kwargs))

    async def async_run(self, **kwargs: Any) -> str:
        """
        Async execution — preferred from the async agent pipeline.
        Calls the MCP client directly without any thread bridge.
        """
        return await self._client.call_tool(self._mcp_name, kwargs)

    def to_model_specific(self, model_provider: ModelProvider) -> dict[str, Any]:
        """
        Convert this tool's schema to the format expected by a specific
        LLM provider.

        MCP tools already carry JSON Schema in ``inputSchema``, which is
        the same format OpenAI's ``tools`` parameter expects — no extra
        transformation is needed beyond wrapping it in the function envelope.
        """
        match model_provider:
            case ModelProvider.OPENAI:
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": self.input_schema,
                    },
                }
            case ModelProvider.ANTHROPIC:
                raise NotImplementedError(
                    "Anthropic tool conversion is not implemented yet."
                )
            case ModelProvider.GEMINI:
                raise NotImplementedError(
                    "Gemini tool conversion is not implemented yet."
                )
            case _:
                raise ValueError(f"Unsupported model provider: {model_provider}")

    def __repr__(self) -> str:
        return (
            f"MCPTool(name='{self.name}', server='{self._server_name}', "
            f"inputs={list(self.input_schema.get('properties', {}))})"
        )
