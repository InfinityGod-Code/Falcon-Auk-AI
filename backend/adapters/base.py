from abc import ABC, abstractmethod
from typing import Any, Optional

from backend.core.base.models.model import ModelProvider
from backend.messages.base_message import BaseMessage
from backend.messages.tool_call import ToolCall
from backend.messages.usage import Usage
from backend.tool_registry import ToolRegistry


class NormalizedResponse:
    def __init__(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
        usage: Optional[Usage] = None,
        finish_reason: Optional[str] = None,
        raw_response: Any = None,
        provider: Optional[ModelProvider] = None,
    ):
        self.content = content
        self.tool_calls = tool_calls
        self.usage = usage or Usage()
        self.finish_reason = finish_reason
        self.raw_response = raw_response
        self.provider = provider


class ProviderAdapter(ABC):
    @abstractmethod
    def normalize_response(self, raw_response: Any) -> NormalizedResponse: ...

    @abstractmethod
    def convert_tools(self, tools: ToolRegistry) -> list[dict]: ...

    @abstractmethod
    def convert_messages(self, messages: list[BaseMessage]) -> list[dict]: ...
