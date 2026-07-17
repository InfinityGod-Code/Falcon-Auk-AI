"""
MCPClient — manages a single MCP server connection with full lifecycle.

Pure-async wrapper around the MCP v1.x ``ClientSession``.  Unlike the
earlier sync-bridge version, this class exposes only ``async`` methods
and runs directly on the caller's event loop — no background thread,
no ``run_coroutine_threadsafe`` dance.

Usage
-----
    client = MCPClient("github", {"type": "stdio", "command": "npx", ...})
    await client.connect()

    tools = await client.list_tools()
    result = await client.call_tool("search", {"query": "..."})

    await client.disconnect()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Raised when the MCP client cannot establish or maintain a connection."""


class MCPToolError(Exception):
    """Raised when an MCP tool call returns an error result."""


class MCPTimeoutError(Exception):
    """Raised when an MCP tool call exceeds the allowed timeout."""


class MCPClient:
    """
    Pure-async wrapper around an MCP ``ClientSession``.

    Call ``await connect()`` to start the session, then use the
    ``call_tool()``, ``list_tools()``, etc. helpers.  Always call
    ``await disconnect()`` to cleanly shut down the transport.

    Parameters
    ----------
    server_name:
        Human-readable label used in log messages.
    transport_config:
        Dictionary describing how to reach the server::

            {"type": "stdio", "command": "npx",
             "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}

            {"type": "sse", "url": "http://localhost:8000/sse"}

            {"type": "http", "url": "https://example.com/mcp",
             "headers": {"Authorization": "Bearer <token>"}}
    """

    def __init__(self, server_name: str, transport_config: dict) -> None:
        self.server_name = server_name
        self.config = transport_config

        self._session: ClientSession | None = None
        self._session_cm: Any = None
        self._transport_cm: Any = None
        self._http_client: httpx.AsyncClient | None = None
        self._get_session_id: Any = None

        # Populated after connect()
        self.server_info: types.Implementation | None = None
        self.server_capabilities: types.ServerCapabilities | None = None
        self.protocol_version: str | None = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Establish the transport and initialise the MCP session.

        Raises ``MCPConnectionError`` on failure.
        """
        transport_type = self.config.get("type", "stdio")

        try:
            if transport_type == "stdio":
                params = StdioServerParameters(
                    command=self.config["command"],
                    args=self.config.get("args", []),
                    env=self.config.get("env"),
                )
                self._transport_cm = stdio_client(params)
            elif transport_type == "sse":
                self._transport_cm = sse_client(
                    url=self.config["url"],
                )
            elif transport_type == "http":
                headers = self.config.get("headers")
                if headers:
                    self._http_client = httpx.AsyncClient(headers=headers)
                self._transport_cm = streamable_http_client(
                    url=self.config["url"],
                    http_client=self._http_client,
                    terminate_on_close=self.config.get("terminate_on_close", True),
                )
            else:
                raise MCPConnectionError(
                    f"Unknown transport type '{transport_type}'; expected 'stdio', 'sse', or 'http'"
                )

            transport_ret = await self._transport_cm.__aenter__()
            if transport_type == "http":
                read, write, self._get_session_id = transport_ret
            else:
                read, write = transport_ret
            self._session_cm = ClientSession(read, write)
            self._session = await self._session_cm.__aenter__()

            init = await self._session.initialize()
            self.server_info = init.serverInfo
            self.server_capabilities = init.capabilities
            self.protocol_version = init.protocolVersion

            logger.info(
                "[%s] Connected  server_info=%s", self.server_name, self.server_info
            )

        except Exception:
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """
        Gracefully close the session and transport streams.
        Idempotent — safe to call multiple times.
        """
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except BaseException:
                pass
            self._session_cm = None
            self._session = None
        if self._http_client:
            try:
                await self._http_client.aclose()
            except BaseException:
                pass
            self._http_client = None
        if self._transport_cm:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            except BaseException:
                pass
            self._transport_cm = None

        self._get_session_id = None
        logger.info("[%s] Disconnected", self.server_name)

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    # ── MCP verbs ────────────────────────────────────────────────────

    async def list_tools(self) -> list[types.Tool]:
        """Fetch the list of tools exposed by the server."""
        result: types.ListToolsResult = await self._session.list_tools()
        return result.tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, timeout: int = 30
    ) -> str:
        """
        Call a tool and return the text concatenation of its content blocks.

        Raises ``MCPToolError`` if the server returned ``isError=true``,
        or ``MCPTimeoutError`` on timeout.
        """
        try:
            async with asyncio.timeout(timeout):
                result: types.CallToolResult = await self._session.call_tool(
                    name, arguments=arguments or {}
                )
        except asyncio.TimeoutError:
            raise MCPTimeoutError(
                f"[{self.server_name}] Tool '{name}' timed out after {timeout}s"
            )

        if result.isError:
            error_text = _extract_text(result.content) or "Unknown MCP error"
            raise MCPToolError(f"[{self.server_name}] Tool '{name}': {error_text}")
        return _extract_text(result.content) or ""

    async def list_resources(self) -> list[types.Resource]:
        result = await self._session.list_resources()
        return result.resources

    async def read_resource(self, uri: str) -> str:
        result = await self._session.read_resource(uri)
        return _extract_text([c for c in result.contents if hasattr(c, "text")])

    async def list_prompts(self) -> list[types.Prompt]:
        result = await self._session.list_prompts()
        return result.prompts

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[types.PromptMessage]:
        result = await self._session.get_prompt(name, arguments=arguments or {})
        return result.messages


# ── Helpers ──────────────────────────────────────────────────────────


def _extract_text(content_blocks: list[Any]) -> str:
    """Concatenate ``TextContent`` blocks from an MCP result."""
    parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
    return "\n".join(parts)
