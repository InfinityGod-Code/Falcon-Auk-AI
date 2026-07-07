from typing import Any, Callable, Generator, Optional

from backend.core.base.tools.tool import Tool
from backend.messages.base_message import (
    BaseMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
)
from backend.messages.tool_call import ToolCall
from backend.messages.usage import Usage, UsageAccumulator
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.response import LLMResponse
from backend.tool_runtime_context import ToolRegistry


def _message_from_dict(data: dict) -> BaseMessage:
    """Reconstruct a BaseMessage subclass from a serialized dict."""
    role = data.get("role")
    if role == "system":
        return SystemMessage(content=data.get("content", ""))
    elif role == "user":
        return UserMessage(content=data.get("content", ""))
    elif role == "assistant":
        tool_calls = None
        raw_tool_calls = data.get("tool_calls")
        if raw_tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                    type=tc.get("type", "function"),
                )
                for tc in raw_tool_calls
            ]
        return AssistantMessage(
            content=data.get("content", ""),
            tool_calls=tool_calls,
        )
    elif role == "tool":
        return ToolMessage(
            content=data.get("content", ""),
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name", ""),
        )
    raise ValueError(f"Unknown message role: {role}")


class LLMLifecycle:
    """
    Manages the full lifecycle of an LLM interaction session:
    message history, tool execution, usage tracking, and callbacks.

    Acts as the engine that agents wrap around. Supports three modes:
      - run()             → single-turn generation
      - run_stream()      → streaming generation
      - run_with_tools()  → iterative tool-calling loop
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        usage_accumulator: Optional[UsageAccumulator] = None,
    ):
        self.provider = provider
        self.tools = tools or []
        self.callbacks = provider.callbacks
        self._messages: list[BaseMessage] = []
        self._usage = usage_accumulator or UsageAccumulator()
        self._last_response: Optional[LLMResponse] = None
        self._metadata: dict[str, Any] = {}

        if system_prompt:
            self._messages.append(SystemMessage(content=system_prompt))

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    @property
    def total_usage(self) -> Usage:
        return self._usage.total

    @property
    def last_response(self) -> Optional[LLMResponse]:
        return self._last_response

    def add_message(self, message: BaseMessage):
        self._messages.append(message)

    def add_messages(self, messages: list[BaseMessage]):
        self._messages.extend(messages)

    def clear_history(self, keep_system: bool = True):
        if keep_system:
            self._messages = [m for m in self._messages if isinstance(m, SystemMessage)]
        else:
            self._messages = []

    def run(self, user_input: str, **kwargs) -> LLMResponse:
        self._messages.append(UserMessage(content=user_input))

        self.callbacks.on_generation_start(
            messages=self._messages, tools=self.tools, model=self.provider.model
        )

        try:
            response = self.provider.generate(
                messages=self._messages,
                tools=self.tools or None,
                **kwargs,
            )
        except Exception as e:
            self.callbacks.on_error(error=e)
            raise

        self._usage.add(response.usage)
        self._last_response = response
        self._messages.append(response.message)

        self.callbacks.on_generation_end(response=response, usage=response.usage)

        return response

    def run_stream(
        self, user_input: str, **kwargs
    ) -> Generator[LLMResponse, None, LLMResponse]:
        self._messages.append(UserMessage(content=user_input))

        self.callbacks.on_generation_start(
            messages=self._messages, tools=self.tools, model=self.provider.model
        )

        full_content = ""
        full_tool_calls: Optional[list[ToolCall]] = None
        final_usage: Optional[Usage] = None

        try:
            for chunk in self.provider.generate_stream(
                messages=self._messages,
                tools=self.tools or None,
                **kwargs,
            ):
                if chunk.message.content:
                    full_content += chunk.message.content
                if chunk.message.tool_calls:
                    full_tool_calls = chunk.message.tool_calls
                if chunk.usage:
                    final_usage = chunk.usage
                self.callbacks.on_stream_chunk(chunk=chunk)
                yield chunk
        except Exception as e:
            self.callbacks.on_error(error=e)
            raise

        final_msg = AssistantMessage(content=full_content, tool_calls=full_tool_calls)
        final_response = LLMResponse(message=final_msg, usage=final_usage or Usage())

        self._usage.add(final_response.usage)
        self._last_response = final_response
        self._messages.append(final_msg)

        self.callbacks.on_generation_end(
            response=final_response, usage=final_response.usage
        )

        return final_response

    def run_with_tools(
        self,
        user_input: str,
        tool_executor: Callable[[ToolCall], Any],
        max_iters: int = 5,
        **kwargs,
    ) -> LLMResponse:
        self._messages.append(UserMessage(content=user_input))

        for iteration in range(max_iters):
            self.callbacks.on_generation_start(
                messages=self._messages, tools=self.tools, model=self.provider.model
            )

            try:
                response = self.provider.generate(
                    messages=self._messages,
                    tools=self.tools or None,
                    **kwargs,
                )
            except Exception as e:
                self.callbacks.on_error(error=e)
                raise

            self._usage.add(response.usage)
            self._last_response = response
            self._messages.append(response.message)

            if not response.message.tool_calls:
                self.callbacks.on_generation_end(
                    response=response, usage=response.usage
                )
                return response

            for tc in response.message.tool_calls:
                self.callbacks.on_tool_call(tool_call=tc)
                try:
                    result = tool_executor(tc)
                except Exception as e:
                    self.callbacks.on_error(error=e)
                    raise
                self._messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc.id,
                        name=tc.function["name"],
                    )
                )
                self.callbacks.on_tool_result(
                    tool_call_id=tc.id, name=tc.function["name"], result=result
                )

        response = self.provider.generate(
            messages=self._messages, tools=self.tools or None, **kwargs
        )
        self._usage.add(response.usage)
        self._last_response = response
        self._messages.append(response.message)
        self.callbacks.on_generation_end(response=response, usage=response.usage)
        return response

    def to_dict(self) -> dict:
        return {
            "provider": self.provider.to_dict(),
            "messages": [m.to_dict() for m in self._messages],
            "metadata": dict(self._metadata),
        }

    @classmethod
    def from_dict(cls, data: dict, provider: BaseLLMProvider) -> "LLMLifecycle":
        instance = cls(provider=provider)
        instance._messages = [_message_from_dict(m) for m in data.get("messages", [])]
        instance._metadata = dict(data.get("metadata", {}))
        return instance

    def reset(self):
        self._messages = []
        self._usage.reset()
        self._last_response = None
        self._metadata = {}
