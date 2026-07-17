"""
Falcon-Auk-AI — Entry point.

Demonstrates:
  1. MCP server integration via Streamable HTTP transport.
  2. MonoAgent execution with MCP tool registry.
  3. Graceful shutdown of MCP connections via MCPSession.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from backend.agents import MonoAgent
from backend.agents.states.agent_state import AgentState
from backend.core.base.base_agent_callback import BaseAgentCallback
from backend.core.base.models.model import ModelProvider
from backend.llm_providers import OpenAILLMProvider
from backend.mcp_integration import MCPSession

console = Console()


class Listener(BaseAgentCallback):
    def state(self, state: AgentState):
        print(f"agent state : {state.current_tool}")

    def logs(self, logs):
        print(f"LOGS FROM AGENT : {logs}")


async def mcp_main():
    PAT = ""
    async with MCPSession() as session:
        registry = await session.add_server(
            name="github",
            transport_config={
                "type": "http",
                "url": "https://api.githubcopilot.com/mcp/x/all",
                "headers": {"Authorization": f"Bearer {PAT}"},
            },
        )
        print(
            f"Registered tools from MCP server: {registry.get_all_schemas(ModelProvider.OPENAI)}"
        )
        mono_agent = MonoAgent(
            provider=OpenAILLMProvider(
                api_key="",
                model="openai/gpt-oss-20b",
                base_url="https://api.groq.com/openai/v1",
            ),
            tools=registry,
            system_prompt="You are helpful assistant.",
            agent_callback=Listener(),
        )
        response = await mono_agent.run("What is most starred repo in Git?")
        print(f"Response from the MCP server : {response.message.content}")


if __name__ == "__main__":
    asyncio.run(mcp_main())
