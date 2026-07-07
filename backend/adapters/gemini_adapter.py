from typing import Any

from backend.adapters.base import NormalizedResponse, ProviderAdapter
from backend.core.base.models.model import ModelProvider
from backend.messages.base_message import BaseMessage
from backend.tool_registry import ToolRegistry


class GeminiAdapter(ProviderAdapter):
    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        raise NotImplementedError("Gemini adapter is not yet implemented.")

    def convert_tools(self, tools: ToolRegistry) -> list[dict]:
        raise NotImplementedError("Gemini adapter is not yet implemented.")

    def convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        raise NotImplementedError("Gemini adapter is not yet implemented.")
