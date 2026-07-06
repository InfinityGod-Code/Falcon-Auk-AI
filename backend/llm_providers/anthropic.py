from typing import Generator, Optional

from backend.core.base.models.model import ModelProvider
from backend.core.base.tools.tool import Tool
from backend.messages.base_message import BaseMessage
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse


class AnthropicLLMProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        callbacks: Optional[CallbackManager] = None,
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, callbacks=callbacks, **kwargs)

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.ANTHROPIC

    def generate(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> LLMResponse:
        raise NotImplementedError("Anthropic provider is not yet implemented.")

    def generate_stream(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> Generator[LLMResponse, None, None]:
        raise NotImplementedError("Anthropic provider is not yet implemented.")

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        raise NotImplementedError("Anthropic provider is not yet implemented.")

    @classmethod
    def from_dict(cls, data: dict, api_key: str) -> "AnthropicLLMProvider":
        return cls(
            api_key=api_key,
            model=data.get("model", "claude-sonnet-4-20250514"),
        )
