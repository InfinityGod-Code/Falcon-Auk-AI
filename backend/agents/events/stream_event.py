"""
Stream event system — used for streaming agent responses.

Each chunk of a stream is wrapped in a typed StreamEvent so
consumers (HTTP SSE, WebSocket, CLI) can react accordingly.
"""

from typing import Any, Optional


class StreamEvent:
    """
    A single event in a streaming response sequence.

    Attributes:
        type: Machine-readable event type ("token", "tool_call", "done", "error").
        data: Payload associated with this stream event.
    """

    def __init__(self, event_type: str, data: dict[str, Any]):
        self.type = event_type
        self.data = data


class TokenStreamEvent(StreamEvent):
    """Yields a partial token of text content."""

    def __init__(self, content: str):
        super().__init__(event_type="token", data={"content": content})


class ToolCallStreamEvent(StreamEvent):
    """Yields a tool call that the assistant wants to execute."""

    def __init__(self, tool_call: Any):
        super().__init__(
            event_type="tool_call",
            data={
                "id": tool_call.id,
                "name": tool_call.function["name"],
                "arguments": tool_call.function["arguments"],
            },
        )


class DoneStreamEvent(StreamEvent):
    """Signals that the stream is complete, with optional usage stats."""

    def __init__(self, usage: Optional[Any] = None):
        super().__init__(
            event_type="done",
            data={
                "usage_tokens": usage.total_tokens if usage else 0,
            },
        )


class ErrorStreamEvent(StreamEvent):
    """Signals that an error occurred during streaming."""

    def __init__(self, error: Exception):
        super().__init__(
            event_type="error",
            data={"error_type": type(error).__name__, "message": str(error)},
        )
