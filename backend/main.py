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

tool_registration = container.tool_registry()
registry = ToolRegistryExecutor(tool_registration)


@registry.tool(
    name="get_weather",
    description="Provides the current weather based on the location.",
)
def get_weather(location: str) -> str:
    return f"The current weather of the {location} is 21 degree celcius and rainy and cold."


console.log(
    f"[bold green]Tool registered: get_weather[/bold green] {tool_registration.get_all_schemas(ModelProvider.OPENAI)}"
)


class Listener(BaseAgentCallback):
    def state(self, state: AgentState):
        print(f"agent state : {state.current_tool}")

    def logs(self, logs):
        print(f"LOGS FROM AGENT : {logs}")


mono_agent = MonoAgent(
    provider=OpenAILLMProvider(
        api_key="",
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
    ),
    tools=tool_registration,
    system_prompt="You are helpful assistant.",
    agent_callback=Listener(),
)

console.log("[bold green]MonoAgent initialized[/bold green]")

user_input = "What is the Weather is Paris ?"
console.log(f'[bold yellow]Sending user input: "{user_input}"[/bold yellow]')

response = mono_agent.run(user_input=user_input)

console.log(f"[bold cyan]Response received: {response.message}[/bold cyan]")
print(
    f"Response from the LLM is here : {response.message} and raw : {response.raw_response} usage : {response.usage.total_tokens}"
)
