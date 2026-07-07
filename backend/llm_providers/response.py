from typing import Any

from backend.messages.base_message import AssistantMessage
from backend.messages.usage import Usage


class LLMResponse:
    def __init__(
        self,
        message: AssistantMessage,
        usage: Usage,
        raw_response: Any = None,
    ):
        self.message = message
        self.usage = usage
        self.raw_response = raw_response
