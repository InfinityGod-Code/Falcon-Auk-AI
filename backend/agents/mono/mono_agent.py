"""
MonoAgent — single-turn agent.

The simplest agent pattern: one user message in → one LLM response out.
No tool loops, no reasoning chains. Wraps LLMLifecycle.run().
"""

from typing import Generator, Optional

from backend.core.base.tools.tool import Tool
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse
from backend.llm_providers.lifecycle import LLMLifecycle
from backend.messages.base_message import SystemMessage, AssistantMessage
from backend.agents.context import ContextManager
from backend.agents.checkpoint import CheckpointManager
from backend.agents.base_agent import BaseAgent
from backend.agents.events.event import CompletionEvent, AgentEvent
from backend.agents.events.stream_event import (
    StreamEvent,
    TokenStreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
)


class MonoAgent(BaseAgent):
    """
    A stateless single-turn agent.

    Each call to run() sends the user input + history to the LLM
    and returns the response. No tool execution loop.

    Example:
        agent = MonoAgent(provider=OpenAILLMProvider(api_key="..."))
        response = agent.run("Hello!")
        print(response.message.content)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Optional[list[Tool]] = None,
        system_prompt: Optional[str] = None,
        callbacks: Optional[CallbackManager] = None,
        name: Optional[str] = None,
        context_manager: Optional[ContextManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        super().__init__(
            provider,
            tools,
            system_prompt,
            callbacks,
            name,
            context_manager=context_manager,
            checkpoint_manager=checkpoint_manager,
        )
        self._lifecycle = LLMLifecycle(
            provider=provider,
            tools=tools,
        )
        if system_prompt:
            self._lifecycle.add_message(SystemMessage(content=system_prompt))

    def run(self, user_input: str, **kwargs) -> LLMResponse:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        response = self._lifecycle.run(user_input, **kwargs)

        self._usage.add(response.usage)
        self._messages.append(self._lifecycle.messages[-2])  # user message
        self._messages.append(self._lifecycle.messages[-1])  # assistant message

        self.emit(CompletionEvent(response.message.content, response.usage, self.name))

        return response

    def run_stream(
        self, user_input: str, **kwargs
    ) -> Generator[StreamEvent, None, LLMResponse]:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        try:
            generator = self._lifecycle.run_stream(user_input, **kwargs)
            final = None
            for chunk in generator:
                if chunk.message.content:
                    event = TokenStreamEvent(chunk.message.content)
                    yield event
                final = chunk

            response = final or LLMResponse(
                message=AssistantMessage(content=""),
                usage=self._lifecycle.total_usage,
            )

            self._usage.add(response.usage)
            self._messages.append(self._lifecycle.messages[-2])
            self._messages.append(self._lifecycle.messages[-1])

            yield DoneStreamEvent(response.usage)
            self.emit(
                CompletionEvent(response.message.content, response.usage, self.name)
            )

            return response

        except Exception as e:
            err_event = ErrorStreamEvent(e)
            yield err_event
            self.emit(AgentEvent("error", {"message": str(e)}, self.name))
            raise
