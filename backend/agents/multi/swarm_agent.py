"""
SwarmAgent — parallel debate / collaboration pattern.

Multiple agents receive the same prompt independently, generate
responses, then a synthesizer (typically an LLM call) merges the
results into a single coherent answer.

Useful for fact-checking, diverse perspective generation, and
tasks that benefit from multiple "points of view".
"""

from typing import Any, Generator, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.core.base.tools.tool import Tool
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.lifecycle import LLMLifecycle
from backend.llm_providers.response import LLMResponse
from backend.messages.base_message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
)
from backend.messages.usage import UsageAccumulator
from backend.agents.base_agent import BaseAgent
from backend.agents.events.event import AgentEvent, CompletionEvent
from backend.agents.events.stream_event import (
    StreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
)
from backend.agents.memory_context import MemoryContextManager
from backend.agents.checkpoint import CheckpointManager
from backend.agents.multi.multi_agent import MultiAgent

_SYNTHESIS_PROMPT = """You are a synthesis agent. Multiple specialist agents have provided
their responses to the same request. Your job is to merge them into
a single coherent, comprehensive answer.

Responses from agents:
{responses}

Provide a unified final answer that captures the best insights from each.
"""


class SwarmAgent(MultiAgent):
    """
    Parallel swarm agent that runs all sub-agents on the same input
    and synthesises their responses.

    Supports both sequential and parallel (thread-pool) execution.

    Example:
        swarm = SwarmAgent(
            provider=OpenAILLMProvider(api_key="..."),
            agents={
                "analyst": AnalystAgent(...),
                "creative": CreativeAgent(...),
            },
        )
        response = swarm.run("What are the implications of AGI?")
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
        parallel: bool = True,
        max_workers: int = 4,
        synthesis_prompt: Optional[str] = None,
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
        self._parallel = parallel
        self._max_workers = max_workers
        self._synthesis_prompt_template = synthesis_prompt or _SYNTHESIS_PROMPT
        if lifecycle is not None:
            self._lifecycle = lifecycle
        else:
            self._lifecycle = LLMLifecycle(provider=provider, tools=None)

    def _run_agents_sequential(self, user_input: str, **kwargs) -> dict[str, str]:
        results = {}
        for name, agent in self._agents.items():
            resp = agent.run(user_input, **kwargs)
            results[name] = resp.message.content or ""
            self._usage.add(resp.usage)
        return results

    def _run_agents_parallel(self, user_input: str, **kwargs) -> dict[str, str]:
        results = {}

        def _run(name: str, agent: BaseAgent) -> tuple[str, str]:
            resp = agent.run(user_input, **kwargs)
            self._usage.add(resp.usage)
            return name, resp.message.content or ""

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(_run, name, agent): name
                for name, agent in self._agents.items()
            }
            for future in as_completed(futures):
                name, content = future.result()
                results[name] = content

        return results

    def _synthesize(self, results: dict[str, str], **kwargs) -> LLMResponse:
        formatted = "\n\n".join(
            f"=== {name} ===\n{content}" for name, content in results.items()
        )
        synthesis_question = self._synthesis_prompt_template.format(responses=formatted)
        self._lifecycle.add_message(UserMessage(content=synthesis_question))

        return self._lifecycle.run(synthesis_question, **kwargs)

    def run(self, user_input: str, **kwargs) -> LLMResponse:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        if self._parallel:
            results = self._run_agents_parallel(user_input, **kwargs)
        else:
            results = self._run_agents_sequential(user_input, **kwargs)

        synthesized = self._synthesize(results, **kwargs)
        self._usage.add(synthesized.usage)

        self.emit(
            CompletionEvent(synthesized.message.content, synthesized.usage, self.name)
        )

        return synthesized

    def run_stream(
        self, user_input: str, **kwargs
    ) -> Generator[StreamEvent, None, LLMResponse]:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))
        try:
            response = self.run(user_input, **kwargs)
            yield DoneStreamEvent(response.usage)
            return response
        except Exception as e:
            yield ErrorStreamEvent(e)
            self.emit(AgentEvent("error", {"message": str(e)}, self.name))
            raise
