"""
ToolCallingAgent — native function-calling agent.

Uses the LLM's built-in tool/function calling API to let the model
decide when to invoke tools. The LLM returns structured tool_call
objects instead of text-based ACTION blocks.

This is the recommended agent for OpenAI / Anthropic models that
support native function calling.
"""

from typing import Any, Callable, Generator, Optional

from backend.core.base.tools.tool import Tool
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.lifecycle import LLMLifecycle
from backend.llm_providers.response import LLMResponse
from backend.messages.base_message import BaseMessage
from backend.agents.base_agent import BaseAgent
from backend.agents.events.event import (
    AgentEvent,
    CompletionEvent,
)
from backend.agents.events.stream_event import (
    StreamEvent,
    TokenStreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
)


class ToolCallingAgent(BaseAgent):
    """
    Agent that uses native tool/function calling.

    Wraps LLMLifecycle.run_with_tools() which handles the full
    tool-calling loop: generate → parse tool_calls → execute →
    feed results back → repeat until no more tool calls.

    Example:
        agent = ToolCallingAgent(
            provider=OpenAILLMProvider(api_key="..."),
            tools=[weather_tool, search_tool],
        )
        response = agent.run_with_tools(
            "What is the weather in London?",
            tool_executor=my_executor,
        )
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Optional[list[Tool]] = None,
        system_prompt: Optional[str] = None,
        callbacks: Optional[CallbackManager] = None,
        name: Optional[str] = None,
    ):
        super().__init__(provider, tools, system_prompt, callbacks, name)
        self._lifecycle = LLMLifecycle(
            provider=provider,
            tools=tools,
            system_prompt=system_prompt,
        )

    def run(self, user_input: str, **kwargs) -> LLMResponse:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        response = self._lifecycle.run(user_input, **kwargs)

        self._usage.add(response.usage)

        self.emit(CompletionEvent(response.message.content, response.usage, self.name))

        return response

    def run_with_tools(
        self,
        user_input: str,
        tool_executor: Callable[[Any], Any],
        max_iters: int = 5,
        **kwargs,
    ) -> LLMResponse:
        """
        Run the agent with automatic tool execution.

        Args:
            user_input:     The user's request.
            tool_executor:  Callable that receives a ToolCall and returns a result.
            max_iters:      Maximum number of tool-calling iterations.

        Returns:
            Final LLMResponse after all tool calls complete.
        """
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        response = self._lifecycle.run_with_tools(
            user_input=user_input,
            tool_executor=tool_executor,
            max_iters=max_iters,
            **kwargs,
        )

        self._usage.add(response.usage)

        self.emit(CompletionEvent(response.message.content, response.usage, self.name))

        return response

    def run_stream(
        self, user_input: str, **kwargs
    ) -> Generator[StreamEvent, None, LLMResponse]:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        try:
            for chunk in self._lifecycle.run_stream(user_input, **kwargs):
                if chunk.message.content:
                    yield TokenStreamEvent(chunk.message.content)

            response = self._lifecycle.last_response

            if response:
                self._usage.add(response.usage)
                self.emit(
                    CompletionEvent(response.message.content, response.usage, self.name)
                )
                yield DoneStreamEvent(response.usage)
                return response

        except Exception as e:
            yield ErrorStreamEvent(e)
            self.emit(AgentEvent("error", {"message": str(e)}, self.name))
            raise

        from backend.messages.base_message import AssistantMessage

        fallback = AssistantMessage(content="")
        fallback_response = LLMResponse(message=fallback, usage=self._usage.total)
        yield DoneStreamEvent(fallback_response.usage)
        return fallback_response
