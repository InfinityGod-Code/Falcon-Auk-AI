"""
MonoAgent — single-turn agent.

The simplest agent pattern: one user message in → one LLM response out.
No tool loops, no reasoning chains. Wraps LLMLifecycle.run().
"""

from typing import Generator, List, Optional
from backend.core.base.base_agent_callback import BaseAgentCallback
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse
from backend.llm_providers.lifecycle import LLMLifecycle
from backend.messages.base_message import (
    BaseMessage,
    SystemMessage,
    AssistantMessage,
    UserMessage,
)
from backend.agents.memory_context import MemoryContextManager
from backend.agents.checkpoint import CheckpointManager
from backend.agents.base_agent import BaseAgent
from backend.agents.events.event import CompletionEvent, AgentEvent
from backend.agents.events.stream_event import (
    StreamEvent,
    TokenStreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
)
from backend.tool_registry import ToolRegistry


class MonoAgent(BaseAgent):
    """
    A stateless single-turn agent.

    Each call to run() sends the user input + history to the LLM
    and returns the response. Tool execution loop untill LLM returns a FINAL_ANSWER without any tool calls.

    Example:
        agent = MonoAgent(provider=OpenAILLMProvider(api_key="..."))
        response = agent.run("Hello!")
        print(response.message.content)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        callbacks: Optional[CallbackManager] = None,
        name: Optional[str] = None,
        context_manager: Optional[MemoryContextManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        agent_callback: Optional[BaseAgentCallback] = None,
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
        self.agent_callback = agent_callback
        if system_prompt:
            self._lifecycle.add_message(SystemMessage(content=system_prompt))

    def run(self, user_input: str, max_iters: int = 10, **kwargs) -> LLMResponse:
        from backend.agents.tool_runner import ToolRunner
        from backend.agents.states.agent_state import AgentState
        from backend.messages.base_message import ToolMessage

        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        executor = ToolRunner(self.tools) if self.tools else None

        messages: List[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            UserMessage(content=user_input),
        ]

        def _cb(**overrides):
            if not self.agent_callback:
                return
            kwargs = dict(
                total_token_consumed=None,
                current_tool=None,
                message_length=len(messages),
                retry_count=None,
                tool_name=None,
            )
            kwargs.update(overrides)
            self.agent_callback.state(AgentState(**kwargs))

        def _log(msg: str):
            if self.agent_callback:
                self.agent_callback.logs(msg)

        _cb(total_token_consumed=0, retry_count=0)
        _log("[MonoAgent] Run started")

        for i in range(max_iters):
            response = self.provider.generate(
                messages=messages, tool_runtime_context=self.tools
            )

            messages.append(response.message)
            tokens = response.usage.total_tokens if response.usage else 0

            if not response.message.tool_calls:
                _cb(total_token_consumed=tokens, retry_count=i)
                _log("[MonoAgent] LLM responded with final answer")
                return response

            _cb(total_token_consumed=tokens, retry_count=i)
            _log(
                f"[MonoAgent] LLM responded with {len(response.message.tool_calls)} tool call(s)"
            )

            if executor:
                for tc in response.message.tool_calls:
                    _cb(current_tool=tc.function["name"], tool_name=tc.function["name"])
                    _log(f"[MonoAgent] Executing tool: {tc.function['name']}")

                    result = executor.execute(tc)
                    messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tc.id,
                            name=tc.function["name"],
                        )
                    )

                    _log(
                        f"[MonoAgent] Tool '{tc.function['name']}' returned: {result[:60]}"
                    )

        _cb(retry_count=max_iters)
        _log(
            f"[MonoAgent] Max iterations ({max_iters}) reached, returning last response"
        )
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
