from typing import Generator, Optional

from backend.core.base.models.model import ModelProvider
from backend.core.base.tools.tool import Tool
from backend.messages.base_message import (
    AssistantMessage,
    BaseMessage,
    ToolMessage,
)
from backend.messages.tool_call import ToolCall
from backend.messages.usage import Usage
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse


class OpenAILLMProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        callbacks: Optional[CallbackManager] = None,
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, callbacks=callbacks, **kwargs)
        self._max_tokens = kwargs.get("max_tokens", 4096)

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OPENAI

    def generate(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key,base_url=kwargs.get("base_url", None))

        raw_messages = [m.to_dict() for m in messages]
        raw_tools = (
            [t.to_model_specific(ModelProvider.OPENAI) for t in tools]
            if tools
            else None
        )

        params = {
            "model": kwargs.get("model", self._model),
            "messages": raw_messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }
        if raw_tools:
            params["tools"] = raw_tools

        response = client.chat.completions.create(**params)
        return self._parse_response(response)

    def generate_stream(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> Generator[LLMResponse, None, None]:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key,base_url=kwargs.get("base_url", None))

        raw_messages = [m.to_dict() for m in messages]
        raw_tools = (
            [t.to_model_specific(ModelProvider.OPENAI) for t in tools]
            if tools
            else None
        )

        params = {
            "model": kwargs.get("model", self._model),
            "messages": raw_messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "stream": True,
        }
        if raw_tools:
            params["tools"] = raw_tools

        stream = client.chat.completions.create(**params)

        full_content = ""
        tool_calls: dict[int, dict] = {}
        usage = None

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                full_content += delta.content
                yield LLMResponse(
                    message=AssistantMessage(content=delta.content),
                    usage=Usage(),
                )

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls[idx]["function"]["name"] += (
                                tc_delta.function.name
                            )
                        if tc_delta.function.arguments:
                            tool_calls[idx]["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )

            if chunk.usage:
                usage = Usage.from_dict(chunk.usage.model_dump())

        final_tool_calls = None
        if tool_calls:
            final_tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
                for tc in sorted(tool_calls.values(), key=lambda x: x["id"])
            ]

        yield LLMResponse(
            message=AssistantMessage(content=full_content, tool_calls=final_tool_calls),
            usage=usage or Usage(),
            raw_response=None,
        )

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self._model)
            text = " ".join(m.content for m in messages)
            return len(encoding.encode(text))
        except ImportError:
            raise NotImplementedError(
                "tiktoken is required for token counting. Install with: pip install tiktoken"
            )

    def get_usage(self, raw_response) -> Usage:
        if raw_response and hasattr(raw_response, "usage") and raw_response.usage:
            return Usage.from_dict(raw_response.usage.model_dump())
        return Usage()

    def get_tool_calls(self, raw_response) -> list[ToolCall]:
        if (
            not raw_response
            or not hasattr(raw_response, "choices")
            or not raw_response.choices
        ):
            return []
        msg = raw_response.choices[0].message
        if not msg.tool_calls:
            return []
        return [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            )
            for tc in msg.tool_calls
        ]

    def _parse_response(self, raw_response) -> LLMResponse:
        choice = raw_response.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id, name=tc.function.name, arguments=tc.function.arguments
                )
                for tc in msg.tool_calls
            ]

        usage = Usage()
        if raw_response.usage:
            usage = Usage(
                prompt_tokens=raw_response.usage.prompt_tokens or 0,
                completion_tokens=raw_response.usage.completion_tokens or 0,
                total_tokens=raw_response.usage.total_tokens or 0,
            )

        assistant_msg = AssistantMessage(
            content=msg.content or "", tool_calls=tool_calls
        )

        return LLMResponse(
            message=assistant_msg, usage=usage, raw_response=raw_response
        )

    @classmethod
    def from_dict(cls, data: dict, api_key: str) -> "OpenAILLMProvider":
        return cls(
            api_key=api_key,
            model=data.get("model", "gpt-4o"),
        )
