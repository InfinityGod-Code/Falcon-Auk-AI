from typing import Any, Optional

from backend.adapters.base import NormalizedResponse, ProviderAdapter
from backend.core.base.models.model import ModelProvider
from backend.messages.base_message import BaseMessage
from backend.messages.tool_call import ToolCall
from backend.messages.usage import Usage
from backend.tool_registry import ToolRegistry


class OpenAIAdapter(ProviderAdapter):
    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        choice = raw_response.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
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

        return NormalizedResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason,
            raw_response=raw_response,
            provider=ModelProvider.OPENAI,
        )

    def convert_tools(self, tools: ToolRegistry) -> list[dict]:
        return tools.get_all_schemas(ModelProvider.OPENAI)

    def convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        return [m.to_dict() for m in messages]
