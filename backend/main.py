"""
Falcon-Auk-AI — Entry point.

Demonstrates:
  1. Local tool registration via @registry.tool decorator.
  2. MCP server integration (stdio / SSE) with tool aggregation.
  3. MonoAgent execution with both local and MCP tools.
  4. Graceful shutdown of MCP connections.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents.states.agent_state import AgentState
from backend.core.base.base_agent_callback import BaseAgentCallback
from backend.core.base.models.model import ModelProvider

# uv run python backend/main.py

from rich.console import Console

from backend.llm_providers.openai import OpenAILLMProvider
from backend.agents.mono.mono_agent import MonoAgent
from backend.core.decorators.tool_registry_executor import ToolRegistryExecutor
from backend.core.dependencies.dependency import container

console = Console()

console.log("[bold blue]Starting Falcon-Auk-AI ...[/bold blue]")

# ────────────────────────────────────────────────────────────────────
#  1.  Tool Registry  (local tools via decorator)
# ────────────────────────────────────────────────────────────────────

tool_registration = container.tool_registry()
registry = ToolRegistryExecutor(tool_registration)


@registry.tool(
    name="get_weather",
    description="Provides the current weather based on the location.",
)
def get_weather(location: str) -> str:
    return f"The current weather of the {location} is 21 degree celcius and rainy and cold."


console.log(
    f"[bold green]Tool registered: get_weather[/bold green] "
    f"{tool_registration.get_all_schemas(ModelProvider.OPENAI)}"
)


# ────────────────────────────────────────────────────────────────────
#  2.  MCP Server Integration
# ────────────────────────────────────────────────────────────────────
#
#  MCPManager is a Singleton that manages connections to one or more
#  MCP servers.  Each server exposes tools that get wrapped as MCPTool
#  instances and registered in the same ToolRegistry as local tools.
#
#  Supported transports:
#    stdio  — launches a local subprocess (e.g. npx-based servers)
#    sse    — connects to a remote HTTP stream endpoint
#
#  Examples (uncomment to use):
#
#      # Filesystem server — list/read/write files
#      manager.add_server("filesystem", {
#          "type": "stdio",
#          "command": "npx",
#          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
#      })
#
#      # GitHub server — search repos, read issues, etc.
#      # Requires GITHUB_TOKEN env variable
#      # manager.add_server("github", {
#      #     "type": "stdio",
#      #     "command": "npx",
#      #     "args": [
#      #         "-y",
#      #         "@modelcontextprotocol/server-github",
#      #     ],
#      #     "env": {"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", "")},
#      # })
#
#      # Remote SSE server
#      # manager.add_server("remote", {
#      #     "type": "sse",
#      #     "url": "http://localhost:8000/sse",
#      # })

mcp_manager = container.mcp_manager()

# ── Connect to Filesystem MCP server (local demo) ───────────────────
try:
    console.log("[bold cyan]Connecting to MCP filesystem server...[/bold cyan]")

    mcp_manager.add_server(
        "filesystem",
        {
            "type": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(Path.home() / "Desktop"),
            ],
        },
    )

    console.log("[bold green]✓ MCP filesystem server connected[/bold green]")

    # ── Register MCP tools into the ToolRegistry ────────────────────
    for mcp_tool in mcp_manager.get_all_tools():
        tool_registration.register_tool(mcp_tool)
        console.log(f"[bold green]  MCP tool registered: {mcp_tool.name}[/bold green]")
    
    print(f"tool schema for MCP tools : {tool_registration.get_all_schemas(ModelProvider.OPENAI)}")

except Exception as exc:
    console.log(f"[bold red]✗ Failed to connect MCP server: {exc}[/bold red]")
    console.log("[bold yellow]  Continuing with local tools only[/bold yellow]")


# ────────────────────────────────────────────────────────────────────
#  3.  Agent Setup
# ────────────────────────────────────────────────────────────────────


class Listener(BaseAgentCallback):
    def state(self, state: AgentState):
        print(f"agent state : {state.current_tool}")

    def logs(self, logs):
        print(f"LOGS FROM AGENT : {logs}")


mono_agent = MonoAgent(
    provider=OpenAILLMProvider(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
    ),
    tools=tool_registration,
    system_prompt="You are helpful assistant.",
    agent_callback=Listener(),
)

console.log("[bold green]MonoAgent initialized[/bold green]")

# ────────────────────────────────────────────────────────────────────
#  4.  Run
# ────────────────────────────────────────────────────────────────────

user_input = "What is the Weather is Paris ?"
console.log(f'[bold yellow]Sending user input: "{user_input}"[/bold yellow]')

try:
    response = mono_agent.run(user_input=user_input)

    console.log(f"[bold cyan]Response received: {response.message}[/bold cyan]")
    print(
        f"Response from the LLM is here : {response.message} "
        f"and raw : {response.raw_response} "
        f"usage : {response.usage.total_tokens}"
    )
finally:
    # ────────────────────────────────────────────────────────────────
    #  5.  Graceful Shutdown
    # ────────────────────────────────────────────────────────────────
    #  Always clean up MCP connections — closes transport streams,
    #  stops event-loop threads, and releases any subprocess.
    # ────────────────────────────────────────────────────────────────
    console.log("[bold yellow]Shutting down MCP connections...[/bold yellow]")
    mcp_manager.shutdown()
    console.log("[bold green]Done.[/bold green]")
