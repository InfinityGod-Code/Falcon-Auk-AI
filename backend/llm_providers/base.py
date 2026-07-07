from abc import ABC, abstractmethod
from typing import Generator, Optional
from backend.core.base.models.model import ModelProvider
from backend.messages.base_message import BaseMessage
from backend.messages.tool_call import ToolCall
from backend.messages.usage import Usage
from backend.llm_providers.callback import BaseCallbackHandler, CallbackManager
from backend.llm_providers.response import LLMResponse
from backend.tool_registry import ToolRegistry


class BaseLLMProvider(ABC):
    def __init__(
        self,
        api_key: str,
        model: str,
        callbacks: Optional[CallbackManager] = None,
        **kwargs,
    ):
        self._api_key = api_key
        self._model = model
        self.callbacks = callbacks or CallbackManager()

    @property
    def model(self) -> str:
        return self._model

    @property
    @abstractmethod
    def provider(self) -> ModelProvider:
        pass

    @abstractmethod
    def generate(
        self,
        messages: list[BaseMessage],
        tool_runtime_context: Optional[ToolRegistry] = None,
        **kwargs,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def generate_stream(
        self,
        messages: list[BaseMessage],
        tool_runtime_context: Optional[ToolRegistry] = None,
        **kwargs,
    ) -> Generator[LLMResponse, None, None]:
        pass


    @abstractmethod
    def count_tokens(self, messages: list[BaseMessage]) -> int: ...

    def get_usage(self, raw_response) -> Usage:
        raise NotImplementedError

    def get_tool_calls(self, raw_response) -> list[ToolCall]:
        raise NotImplementedError

    def add_callback_handler(self, handler: BaseCallbackHandler):
        self.callbacks.add_handler(handler)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider.value,
            "model": self._model,
        }

    @classmethod
    def from_dict(cls, data: dict, api_key: str) -> "BaseLLMProvider":
        raise NotImplementedError
