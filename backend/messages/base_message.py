from abc import ABC, abstractmethod
from typing import Optional

from backend.messages.roles import MessageRole
from backend.messages.tool_call import ToolCall


class BaseMessage(ABC):
    def __init__(self, content: str, role: MessageRole):
        self.content = content
        self.role = role

    @abstractmethod
    def to_dict(self) -> dict: ...


class SystemMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(content=content, role=MessageRole.SYSTEM)

    def to_dict(self) -> dict:
        return {"role": "system", "content": self.content}


class UserMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(content=content, role=MessageRole.USER)

    def to_dict(self) -> dict:
        return {"role": "user", "content": self.content}


class AssistantMessage(BaseMessage):
    def __init__(self, content: str, tool_calls: Optional[list[ToolCall]] = None):
        super().__init__(content=content, role=MessageRole.ASSISTANT)
        self.tool_calls = tool_calls or []

    def to_dict(self) -> dict:
        d = {"role": "assistant"}
        if self.tool_calls:
            d["content"] = None
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        else:
            d["content"] = self.content
        return d


class ToolMessage(BaseMessage):
    def __init__(self, content: str, tool_call_id: str, name: str):
        super().__init__(content=content, role=MessageRole.TOOL)
        self.tool_call_id = tool_call_id
        self.name = name

    def to_dict(self) -> dict:
        return {
            "role": "tool",
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }
