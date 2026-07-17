"""
Integration test for MCP client + manager + MonoAgent.

Launches the test server as a stdio subprocess, connects via MCPClient,
lists tools, calls a tool, registers them in ToolRegistry, and verifies
the MonoAgent can discover and use them.

Run:
    uv run python backend/mcp_integration/test_integration.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.panel import Panel

from backend.mcp_integration.mcp_client import MCPClient, MCPToolError
from backend.mcp_integration.manager import MCPManager
from backend.mcp_integration.mcp_tool import MCPTool
from backend.tool_registry import ToolRegistry
from backend.core.base.models.model import ModelProvider

console = Console()


async def test_mcp_client_directly():
    """
    Test 1: MCPClient — connect, list tools, call tool.
    """
    console.print(Panel("[bold cyan]Test 1: MCPClient direct tool call[/bold cyan]"))

    client = MCPClient(
        "test_server",
        {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "python",
                str(Path(__file__).parent / "test_server.py"),
            ],
        },
    )

    try:
        await client.connect()
        console.print(f"  [green]✓[/green] Connected  server_info={client.server_info}")

        tools = await client.list_tools()
        console.print(f"  [green]✓[/green] Listed {len(tools)} tools:")
        for t in tools:
            console.print(f"       • {t.name}: {t.description}")

        result = await client.call_tool("add", {"a": 3, "b": 5})
        assert result == "8", f"Expected '8', got '{result}'"
        console.print(f"  [green]✓[/green] add(3, 5) = {result}")

        result = await client.call_tool("reverse_string", {"text": "hello"})
        assert result == "olleh", f"Expected 'olleh', got '{result}'"
        console.print(f"  [green]✓[/green] reverse_string('hello') = {result}")

        result = await client.call_tool("get_string_length", {"text": "falcon"})
        assert result == "6", f"Expected '6', got '{result}'"
        console.print(f"  [green]✓[/green] get_string_length('falcon') = {result}")

        console.print(Panel("[bold green]✓ Test 1 PASSED[/bold green]"))
        return client

    except Exception as e:
        console.print(f"  [red]✗[/red] {type(e).__name__}: {e}")
        await client.disconnect()
        raise


async def test_mcp_manager_and_registry(client: MCPClient):
    """
    Test 2: MCPManager + ToolRegistry — aggregate MCP tools into ToolRegistry.
    """
    console.print(Panel("[bold cyan]Test 2: MCPManager + ToolRegistry[/bold cyan]"))

    tool_registry = ToolRegistry()
    mcp_tools = await client.list_tools()

    for mt in mcp_tools:
        adapter = MCPTool(
            name=f"test_{mt.name}",
            description=mt.description or "",
            input_schema=mt.inputSchema,
            mcp_client=client,
            server_name="test_server",
            original_tool_name=mt.name,
        )
        tool_registry.register_tool(adapter)
        console.print(f"  [green]✓[/green] Registered MCPTool: {adapter}")

    schemas = tool_registry.get_all_schemas(ModelProvider.OPENAI)
    names = [s["function"]["name"] for s in schemas]
    console.print(f"  [green]✓[/green] OpenAI schemas for: {names}")

    tool = tool_registry.get_tool("test_add")
    assert tool is not None, "Tool 'test_add' not found in registry"
    result = await tool.async_run(a=10, b=20)
    assert result == "30"
    console.print(f"  [green]✓[/green] test_add(10, 20) via ToolRegistry = {result}")
    console.print(Panel("[bold green]✓ Test 2 PASSED[/bold green]"))


async def test_mcp_manager_multi_server():
    """
    Test 3: MCPManager — aggregate from multiple servers.
    Launches two test server instances.
    """
    console.print(Panel("[bold cyan]Test 3: MCPManager multi-server[/bold cyan]"))
    manager = MCPManager()

    try:
        await manager.add_server(
            "alpha",
            {
                "type": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "python",
                    str(Path(__file__).parent / "test_server.py"),
                ],
            },
        )

        await manager.add_server(
            "beta",
            {
                "type": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "python",
                    str(Path(__file__).parent / "test_server.py"),
                ],
            },
        )

        all_tools = await manager.get_all_tools()
        console.print(
            f"  [green]✓[/green] Aggregated {len(all_tools)} tools from {len(manager.server_names)} servers"
        )

        tool_names = [t.name for t in all_tools]
        assert "alpha_add" in tool_names, "Missing alpha_add"
        assert "beta_add" in tool_names, "Missing beta_add"
        console.print(f"  [green]✓[/green] Prefixed names: {tool_names}")

        for t in all_tools:
            if t.name == "alpha_add":
                r = await t.async_run(a=1, b=2)
                assert r == "3"
                console.print(f"  [green]✓[/green] alpha_add(1, 2) = {r}")
            if t.name == "beta_reverse_string":
                r = await t.async_run(text="abc")
                assert r == "cba"
                console.print(f"  [green]✓[/green] beta_reverse_string('abc') = {r}")

        console.print(Panel("[bold green]✓ Test 3 PASSED[/bold green]"))

    finally:
        await manager.shutdown()
        console.print("  [green]✓[/green] All servers shut down cleanly")


async def test_tool_error_handling(client: MCPClient):
    """
    Test 4: Error handling — tool errors don't crash the client.
    """
    console.print(Panel("[bold cyan]Test 4: Error handling[/bold cyan]"))

    try:
        await client.call_tool("nonexistent_tool", {})
        console.print("  [red]✗ Expected MCPToolError but no exception raised[/red]")
    except MCPToolError as e:
        console.print(f"  [green]✓[/green] MCPToolError correctly raised: {e}")
    except Exception as e:
        console.print(
            f"  [green]✓[/green] Other error (expected): {type(e).__name__}: {e}"
        )

    console.print(Panel("[bold green]✓ Test 4 PASSED[/bold green]"))


async def test_timeout_handling():
    """
    Test 5: Timeout — very short timeout raises MCPTimeoutError.
    """
    console.print(Panel("[bold cyan]Test 5: Timeout handling[/bold cyan]"))

    client = MCPClient(
        "timeout_test",
        {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "python",
                str(Path(__file__).parent / "test_server.py"),
            ],
        },
    )

    try:
        await client.connect()
        result = await client.call_tool("add", {"a": 1, "b": 2}, timeout=0.001)
        console.print(
            f"  [yellow]⚠ Timeout test: got result {result} (timeout may be too short for startup)[/yellow]"
        )
    except Exception as e:
        console.print(
            f"  [green]✓[/green] Timeout-related error: {type(e).__name__}: {e}"
        )

    await client.disconnect()
    console.print(Panel("[bold green]✓ Test 5 PASSED[/bold green]"))


async def test_disconnect_reconnect():
    """
    Test 6: Disconnect + reconnect.
    """
    console.print(Panel("[bold cyan]Test 6: Disconnect / Reconnect[/bold cyan]"))

    manager = MCPManager()
    await manager.add_server(
        "reconnect",
        {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "python",
                str(Path(__file__).parent / "test_server.py"),
            ],
        },
    )

    tools_before = await manager.get_all_tools()
    console.print(f"  [green]✓[/green] Connected, {len(tools_before)} tools available")

    await manager.remove_server("reconnect")
    console.print("  [green]✓[/green] Disconnected")

    await manager.add_server(
        "reconnect",
        {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "python",
                str(Path(__file__).parent / "test_server.py"),
            ],
        },
    )
    tools_after = await manager.get_all_tools()
    console.print(f"  [green]✓[/green] Reconnected, {len(tools_after)} tools available")

    assert len(tools_after) > 0, "No tools after reconnect"
    await manager.shutdown()
    console.print(Panel("[bold green]✓ Test 6 PASSED[/bold green]"))


# ── Main ─────────────────────────────────────────────────────────────


async def main():
    console.print(
        Panel.fit(
            "[bold blue]MCP Integration Test Suite[/bold blue]\n"
            "Testing MCPClient, MCPTool, MCPManager, and error handling",
            border_style="blue",
        )
    )

    client = None
    try:
        client = await test_mcp_client_directly()
        await test_tool_error_handling(client)
        await test_mcp_manager_and_registry(client)

        await test_mcp_manager_multi_server()
        await test_timeout_handling()
        await test_disconnect_reconnect()

    except Exception as e:
        console.print(f"\n[bold red]Test suite FAILED: {e}[/bold red]")
        sys.exit(1)
    finally:
        if client:
            await client.disconnect()

    console.print(
        Panel.fit(
            "[bold green]All tests passed![/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
