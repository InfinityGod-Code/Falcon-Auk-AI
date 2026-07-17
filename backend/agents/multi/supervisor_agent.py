"""
SupervisorAgent — a "boss" agent that delegates to specialized sub-agents.

The supervisor LLM decides which sub-agent to invoke based on the
user's request, passes the task, and collects the result. This
enables modular specialization (e.g., a coder_agent, searcher_agent,
calculator_agent).

Optionally supports multiple rounds of delegation.
"""

from typing import Any, AsyncGenerator, Optional

from backend.core.base.tools.tool import Tool
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.lifecycle import LLMLifecycle
from backend.llm_providers.response import LLMResponse
from backend.messages.base_message import SystemMessage, UserMessage, AssistantMessage
from backend.messages.usage import UsageAccumulator
from backend.agents.base_agent import BaseAgent
from backend.agents.events.event import AgentEvent, CompletionEvent
from backend.agents.events.stream_event import (
    StreamEvent,
    TokenStreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
)
from backend.agents.memory_context import MemoryContextManager
from backend.agents.checkpoint import CheckpointManager
from backend.agents.multi.multi_agent import MultiAgent

_SUPERVISOR_PROMPT = """You are a supervisor agent. Your job is to decide which specialist
agent should handle the user's request, delegate to it, and return the result.

Available agents:
{agent_list}

Respond with EXACTLY ONE of:
  DELEGATE: <agent_name>
  TASK: <task description for the agent>
  — or —
  FINAL: <your direct answer>
"""


class SupervisorAgent(MultiAgent):
    """
    Supervisor delegates tasks to named sub-agents.

    The supervisor LLM decides which agent to invoke and what
    task to give it. The sub-agent's response is returned to the user.

    Example:
        supervisor = SupervisorAgent(
            provider=OpenAILLMProvider(api_key="..."),
            agents={
                "coder": CodingAgent(...),
                "searcher": SearchAgent(...),
            },
        )
        response = supervisor.run("Write a Python script to fetch weather data.")
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Optional[list[Tool]] = None,
        system_prompt: Optional[str] = None,
        callbacks: Optional[CallbackManager] = None,
        name: Optional[str] = None,
        context_manager: Optional[MemoryContextManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        agents: Optional[dict[str, BaseAgent]] = None,
        max_delegations: int = 3,
        lifecycle: Optional[LLMLifecycle] = None,
        usage_accumulator: Optional[UsageAccumulator] = None,
    ):
        super().__init__(
            provider,
            tools,
            system_prompt,
            callbacks,
            name,
            context_manager=context_manager,
            checkpoint_manager=checkpoint_manager,
            agents=agents,
            usage_accumulator=usage_accumulator,
        )
        self._max_delegations = max_delegations

        agents_list = "\n".join(
            f"  {name}: {type(agent).__name__}"
            for name, agent in (agents or {}).items()
        )
        prompt = system_prompt or _SUPERVISOR_PROMPT.format(agent_list=agents_list)

        if lifecycle is not None:
            self._lifecycle = lifecycle
        else:
            self._lifecycle = LLMLifecycle(provider=provider, tools=None)
        self._lifecycle.add_message(SystemMessage(content=prompt))

    def _parse_delegation(self, text: str) -> Optional[tuple[str, str]]:
        """Parse DELEGATE / TASK blocks from supervisor output."""
        import re

        delegate = re.search(r"DELEGATE:\s*(\w+)", text)
        task = re.search(r"TASK:\s*(.+)", text, re.DOTALL)
        if delegate:
            return delegate.group(1).strip(), (task.group(1).strip() if task else "")
        return None

    def _parse_final(self, text: str) -> Optional[str]:
        """Extract FINAL answer if present."""
        import re

        match = re.search(r"FINAL:\s*(.+)", text, re.DOTALL)
        return match.group(1).strip() if match else None

    async def run(self, user_input: str, **kwargs) -> LLMResponse:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        self._lifecycle.add_message(UserMessage(content=user_input))

        for round_num in range(self._max_delegations):
            response = await self._lifecycle.provider.generate(
                messages=self._lifecycle.messages,
                **kwargs,
            )
            self._usage.add(response.usage)
            content = response.message.content or ""
            self._lifecycle.messages.append(response.message)

            final = self._parse_final(content)
            if final:
                self.emit(CompletionEvent(final, response.usage, self.name))
                return LLMResponse(
                    message=AssistantMessage(content=final),
                    usage=response.usage,
                )

            parsed = self._parse_delegation(content)
            if parsed:
                agent_name, task = parsed
                agent = self.get_agent(agent_name)
                if agent:
                    agent_result = await agent.run(task, **kwargs)
                    result_text = agent_result.message.content or ""
                    self._lifecycle.add_message(
                        AssistantMessage(
                            content=f"Agent '{agent_name}' returned:\n{result_text}"
                        )
                    )
                else:
                    self._lifecycle.add_message(
                        AssistantMessage(
                            content=f"Agent '{agent_name}' not found. Available: {self.agent_names}"
                        )
                    )

        response = await self._lifecycle.provider.generate(
            messages=self._lifecycle.messages,
            **kwargs,
        )
        self._usage.add(response.usage)
        self.emit(
            CompletionEvent(response.message.content or "", response.usage, self.name)
        )
        return response

    async def run_stream(
        self, user_input: str, **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))
        try:
            response = await self.run(user_input, **kwargs)
            yield DoneStreamEvent(response.usage)
        except Exception as e:
            yield ErrorStreamEvent(e)
            self.emit(AgentEvent("error", {"message": str(e)}, self.name))
            raise
