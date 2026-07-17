"""
MCPClient — manages a single MCP server connection with full lifecycle.

Connects via stdio (subprocess) or SSE (HTTP stream), wraps the v1.x
ClientSession in a dedicated background thread with its own asyncio
event loop.  Exposes synchronous methods so the rest of the framework
(which is sync) can call MCP tools without refactoring to async.

Thread architecture
-------------------
    main thread                          bg thread (daemon)
    ───────────                          ─────────────────
    MCPTool.run() ──coro_threadsafe──▶   loop.run_forever()
        │                                    │
        │ future.result() ◀─── sets future ──┤
        │                                    │
        │                              ClientSession.call_tool()
        │                                    │
        │                              stdio / SSE → MCP Server

Usage
-----
    client = MCPClient("github", {"type": "stdio", "command": "npx", ...})
    client.connect()

    tools = client.list_tools()       # synchronous
    result = client.call_tool("search", {"query": "..."})

    client.disconnect()               # cleanup
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future, TimeoutError
from threading import Thread
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Raised when the MCP client cannot establish or maintain a connection."""


class MCPToolError(Exception):
    """Raised when an MCP tool call returns an error result."""


class MCPTimeoutError(Exception):
    """Raised when an MCP tool call exceeds the allowed timeout."""


class MCPClient:
    """
    Synchronous wrapper around an MCP ClientSession.

    Runs the asyncio event loop in a dedicated daemon thread so that
    all public methods are synchronous and can be called from our
    synchronous agent pipeline (ToolRunner, MonoAgent, etc.).

    Parameters
    ----------
    server_name:
        Human-readable label used in log messages.
    transport_config:
        Dictionary describing how to reach the server::

            # stdio — launches a subprocess
            {"type": "stdio", "command": "npx",
             "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}

            # SSE — connects to a remote HTTP endpoint
            {"type": "sse", "url": "http://localhost:8000/sse"}
    """

    def __init__(self, server_name: str, transport_config: dict) -> None:
        self.server_name = server_name
        self.config = transport_config

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._session: ClientSession | None = None
        self._session_cm: Any = None  # stashed to prevent GC from calling __aexit__
        self._transport_cm: Any = None
        self._read: Any = None
        self._write: Any = None

        self._connected_future: Future | None = None
        self._disconnect_requested = False

        # Cached after connect()
        self.server_info: types.Implementation | None = None
        self.server_capabilities: types.ServerCapabilities | None = None
        self.protocol_version: str | None = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Start the background event-loop thread and block until the MCP
        session is initialized.

        Raises MCPConnectionError if the connection cannot be established
        within the default timeout (30 s).
        """
        if self._thread and self._thread.is_alive():
            logger.warning(
                "[%s] Already connected, ignoring duplicate connect()", self.server_name
            )
            return

        self._disconnect_requested = False
        self._connected_future = Future()

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(
            target=self._run_event_loop,
            name=f"mcp-{self.server_name}",
            daemon=True,
        )
        self._thread.start()

        # Block until the async connection routine completes (or times out)
        try:
            self._connected_future.result(timeout=30)
        except TimeoutError:
            self.disconnect()
            raise MCPConnectionError(
                f"[{self.server_name}] Connection timed out after 30 s"
            ) from None

        logger.info(
            "[%s] Connected  server_info=%s", self.server_name, self.server_info
        )

    def disconnect(self) -> None:
        """
        Gracefully shut down the MCP session, transport streams, and
        event-loop thread.  Idempotent — safe to call multiple times.
        """
        if not self._loop or self._loop.is_closed():
            return
        self._disconnect_requested = True

        async def _cleanup() -> None:
            try:
                if self._session_cm:
                    await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                if self._transport_cm:
                    await self._transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._loop.stop()

        asyncio.run_coroutine_threadsafe(_cleanup(), self._loop)

        if self._thread:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                logger.warning(
                    "[%s] Cleanup thread did not finish within 10 s", self.server_name
                )

        self._session = None
        self._read = self._write = None
        self._loop = None
        self._thread = None

        logger.info("[%s] Disconnected", self.server_name)

    @property
    def is_connected(self) -> bool:
        """Check whether the client is currently connected."""
        return bool(
            self._session is not None and self._thread and self._thread.is_alive()
        )

    # ── MCP verbs (synchronous public API) ───────────────────────────

    def list_tools(self) -> list[types.Tool]:
        """
        Fetch the list of tools exposed by the server.

        Returns
        -------
        list[types.Tool]
            Each tool carries ``name``, ``description``, and ``inputSchema``.
        """
        return self._run_coro(self._async_list_tools())

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, timeout: int = 30
    ) -> str:
        """
        Call a tool on the MCP server and return its text result.

        Parameters
        ----------
        name:
            Tool name as reported by ``list_tools()``.
        arguments:
            JSON-serialisable dict of keyword arguments.
        timeout:
            Maximum seconds to wait for the server to respond.

        Returns
        -------
        str
            Concatenated text content from the tool result.

        Raises
        ------
        MCPToolError
            If the server returned ``isError=true``.
        MCPTimeoutError
            If the call did not complete within *timeout* seconds.
        """
        return self._run_coro(
            self._async_call_tool(name, arguments or {}, timeout),
            timeout=timeout + 5,
        )

    def list_resources(self) -> list[types.Resource]:
        """Fetch the list of resources exposed by the server."""
        return self._run_coro(self._async_list_resources())

    def read_resource(self, uri: str) -> str:
        """Read a resource by its URI and return the text content."""
        return self._run_coro(self._async_read_resource(uri))

    def list_prompts(self) -> list[types.Prompt]:
        """Fetch the list of prompt templates exposed by the server."""
        return self._run_coro(self._async_list_prompts())

    def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[types.PromptMessage]:
        """Render a prompt template and return the resulting messages."""
        return self._run_coro(self._async_get_prompt(name, arguments or {}))

    # ── Internal: event loop runner ──────────────────────────────────

    def _run_event_loop(self) -> None:
        """Target for the background thread — owns the asyncio loop."""
        asyncio.set_event_loop(self._loop)
        try:
            # Schedule the async connection routine
            asyncio.ensure_future(self._async_connect())
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _async_connect(self) -> None:
        """
        Async connection routine: create transport → open session →
        initialize → cache server metadata.
        """
        try:
            transport_type = self.config.get("type", "stdio")

            if transport_type == "stdio":
                params = StdioServerParameters(
                    command=self.config["command"],
                    args=self.config.get("args", []),
                    env=self.config.get("env"),
                )
                self._transport_cm = stdio_client(params)
                self._read, self._write = await self._transport_cm.__aenter__()
            elif transport_type == "sse":
                self._transport_cm = sse_client(
                    url=self.config["url"],
                    headers=self.config.get("headers"),
                )
                self._read, self._write = await self._transport_cm.__aenter__()
            else:
                raise MCPConnectionError(
                    f"Unknown transport type '{transport_type}'; expected 'stdio' or 'sse'"
                )

            self._session_cm = ClientSession(self._read, self._write)
            self._session = await self._session_cm.__aenter__()
            init = await self._session.initialize()

            self.server_info = init.serverInfo
            self.server_capabilities = init.capabilities
            self.protocol_version = init.protocolVersion

            if self._connected_future and not self._connected_future.done():
                self._connected_future.set_result(True)
        except Exception as exc:
            if self._connected_future and not self._connected_future.done():
                self._connected_future.set_exception(exc)
            else:
                logger.exception("[%s] Connection failed", self.server_name)

    # ── Internal: async MCP verb implementations ─────────────────────

    async def _async_list_tools(self) -> list[types.Tool]:
        result: types.ListToolsResult = await self._session.list_tools()
        return result.tools

    async def _async_call_tool(
        self, name: str, arguments: dict[str, Any], timeout: int
    ) -> str:
        result: types.CallToolResult = await asyncio.wait_for(
            self._session.call_tool(name, arguments=arguments),
            timeout=timeout,
        )
        if result.isError:
            error_text = _extract_text(result.content) or "Unknown MCP error"
            raise MCPToolError(f"[{self.server_name}] Tool '{name}': {error_text}")
        return _extract_text(result.content) or ""

    async def _async_list_resources(self) -> list[types.Resource]:
        result = await self._session.list_resources()
        return result.resources

    async def _async_read_resource(self, uri: str) -> str:
        result = await self._session.read_resource(uri)
        return _extract_text([c for c in result.contents if hasattr(c, "text")])

    async def _async_list_prompts(self) -> list[types.Prompt]:
        result = await self._session.list_prompts()
        return result.prompts

    async def _async_get_prompt(
        self, name: str, arguments: dict[str, str]
    ) -> list[types.PromptMessage]:
        result = await self._session.get_prompt(name, arguments=arguments)
        return result.messages

    # ── Internal: sync ↔ async bridge ────────────────────────────────

    def _run_coro(self, coro, timeout: int | None = None) -> Any:
        """
        Schedule a coroutine on the background event loop and block
        until it finishes (or *timeout* expires).
        """
        if not self._loop or self._loop.is_closed():
            raise MCPConnectionError(f"[{self.server_name}] Not connected")
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            raise MCPTimeoutError(
                f"[{self.server_name}] Operation timed out after {timeout}s"
            ) from None


# ── Helpers ──────────────────────────────────────────────────────────


def _extract_text(content_blocks: list[Any]) -> str:
    """
    Concatenate all ``TextContent`` blocks from an MCP result into a
    single string.
    """
    parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
    return "\n".join(parts)
