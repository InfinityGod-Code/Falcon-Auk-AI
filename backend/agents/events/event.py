"""
Agent event system — Observer pattern.

Agents emit typed events during execution. External handlers
can subscribe via agent.on(event_type, handler) to react to
tool calls, errors, completions, etc.
"""

import time
from typing import Any, Optional


class AgentEvent:
    """
    Base event emitted by agents during run() or run_stream().

    Attributes:
        type:      Machine-readable event type (e.g. "tool_call", "error").
        data:      Arbitrary payload attached to the event.
        agent:     Name of the agent that emitted the event.
        timestamp: Unix timestamp when the event was created.
    """

    def __init__(
        self,
        event_type: str,
        data: dict[str, Any],
        agent: str,
        timestamp: Optional[float] = None,
    ):
        self.type = event_type
        self.data = data
        self.agent = agent
        self.timestamp = timestamp or time.time()


class ToolCallEvent(AgentEvent):
    """Emitted when an agent invokes a tool during execution."""

    def __init__(self, tool_call: Any, agent: str):
        super().__init__(
            event_type="tool_call",
            data={
                "tool_call_id": tool_call.id,
                "name": tool_call.function["name"],
                "arguments": tool_call.function["arguments"],
            },
            agent=agent,
        )


class ToolResultEvent(AgentEvent):
    """Emitted when a tool execution completes and a result is available."""

    def __init__(self, tool_call_id: str, name: str, result: Any, agent: str):
        super().__init__(
            event_type="tool_result",
            data={"tool_call_id": tool_call_id, "name": name, "result": result},
            agent=agent,
        )


class ThoughtEvent(AgentEvent):
    """Emitted during ReAct-style reasoning when the agent produces a thought."""

    def __init__(self, thought: str, agent: str):
        super().__init__(
            event_type="thought",
            data={"thought": thought},
            agent=agent,
        )


class ErrorEvent(AgentEvent):
    """Emitted when an agent encounters a recoverable or fatal error."""

    def __init__(self, error: Exception, agent: str):
        super().__init__(
            event_type="error",
            data={"error_type": type(error).__name__, "message": str(error)},
            agent=agent,
        )


class CompletionEvent(AgentEvent):
    """Emitted when an agent finishes a run() successfully."""

    def __init__(self, content: str, usage: Any, agent: str):
        super().__init__(
            event_type="completion",
            data={
                "content": content,
                "usage_tokens": usage.total_tokens if usage else 0,
            },
            agent=agent,
        )
