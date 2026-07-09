from typing import Generator, Optional
from backend.core.base.models.model import ModelProvider
from backend.messages.base_message import (
    AssistantMessage,
    BaseMessage,
)
from backend.messages.tool_call import ToolCall
from backend.messages.usage import Usage
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse
from rich.console import Console
from openai import OpenAI
from backend.tool_registry import ToolRegistry
from backend.adapters.openai_adapter import OpenAIAdapter

console = Console()


class OpenAILLMProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        callbacks: Optional[CallbackManager] = None,
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, callbacks=callbacks, **kwargs)
        self._base_url = kwargs.get("base_url")
        self._max_tokens = kwargs.get("max_tokens", 4096)
        self._adapter = OpenAIAdapter()

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OPENAI

    def generate(
        self,
        messages: list[BaseMessage],
        tool_registry: Optional[ToolRegistry] = None,
        **kwargs,
    ) -> LLMResponse:
        
        console.log(
            f"[bold green]OpenAI Provider: base URL : {self._base_url}[/bold green]"
        )

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        raw_messages = self._adapter.convert_messages(messages)
        raw_tools = (
            self._adapter.convert_tools(tool_registry)
            if tool_registry
            else None
        )
        console.log(
            f"[bold green]OpenAI Provider: Generating response with model {raw_tools}[/bold green]"
        )

        params = {
            "model": kwargs.get("model", self._model),
            "messages": raw_messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }
        if raw_tools:
            params["tools"] = raw_tools

        response = client.chat.completions.create(**params)
        normalized = self._adapter.normalize_response(response)

        return LLMResponse(
            message=AssistantMessage(
                content=normalized.content, tool_calls=normalized.tool_calls
            ),
            usage=normalized.usage,
            raw_response=response,
        )

    def generate_stream(
        self,
        messages: list[BaseMessage],
        tool_runtime_context: Optional[ToolRegistry] = None,
        **kwargs,
    ) -> Generator[LLMResponse, None, None]:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)

        raw_messages = self._adapter.convert_messages(messages)
        raw_tools = (
            self._adapter.convert_tools(tool_runtime_context)
            if tool_runtime_context
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
        return self._adapter.normalize_response(raw_response).usage

    def get_tool_calls(self, raw_response) -> list[ToolCall]:
        normalized = self._adapter.normalize_response(raw_response)
        return normalized.tool_calls or []

    @classmethod
    def from_dict(cls, data: dict, api_key: str) -> "OpenAILLMProvider":
        return cls(
            api_key=api_key,
            model=data.get("model", "gpt-4o"),
        )
