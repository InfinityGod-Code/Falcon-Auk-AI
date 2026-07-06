"""
BaseAgent — abstract foundation for every agent in the system.

All agents share:
  - A provider (LLM), tools, system prompt, callbacks.
  - Message history management.
  - Usage tracking.
  - An event system (Observer pattern) via emit() / on().
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Generator, Optional

from backend.core.base.tools.tool import Tool
from backend.messages.base_message import (
    BaseMessage,
    SystemMessage,
    AssistantMessage,
)
from backend.messages.usage import Usage, UsageAccumulator
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import BaseCallbackHandler, CallbackManager
from backend.llm_providers.response import LLMResponse
from backend.llm_providers.lifecycle import _message_from_dict
from backend.agents.events.event import AgentEvent


class BaseAgent(ABC):
    """
    Abstract base for all agent types.

    Subclasses must implement:
      - run()              — synchronous execution
      - run_stream()       — streaming execution

    Convention:
      - Subclasses call self.emit() at key lifecycle points.
      - Usage is accumulated via self._usage.
      - Message history lives in self._messages.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Optional[list[Tool]] = None,
        system_prompt: Optional[str] = None,
        callbacks: Optional[CallbackManager] = None,
        name: Optional[str] = None,
    ):
        self.provider = provider
        self.tools = tools or []
        self.callbacks = callbacks or CallbackManager()
        self.system_prompt = system_prompt
        self.name = name or self.__class__.__name__

        self._messages: list[BaseMessage] = []
        self._usage = UsageAccumulator()
        self._event_listeners: dict[str, list[Callable[[AgentEvent], None]]] = (
            defaultdict(list)
        )

        if system_prompt:
            self._messages.append(SystemMessage(content=system_prompt))

    # ── Abstract execution methods ──────────────────────────────────

    @abstractmethod
    def run(self, user_input: str, **kwargs) -> LLMResponse:
        """Execute the agent on a single user input and return a response."""
        ...

    @abstractmethod
    def run_stream(
        self, user_input: str, **kwargs
    ) -> Generator[Any, None, LLMResponse]:
        """
        Execute the agent in streaming mode.

        Yields stream events (tokens, tool_calls, etc.)
        and returns the final LLMResponse.
        """
        ...

    # ── Observer pattern ────────────────────────────────────────────

    def emit(self, event: AgentEvent):
        """
        Emit an event to all registered listeners.

        Supports wildcard "*" listeners that receive every event.
        """
        for handler in self._event_listeners.get(event.type, []):
            handler(event)
        for handler in self._event_listeners.get("*", []):
            handler(event)

    def on(self, event_type: str, handler: Callable[[AgentEvent], None]):
        """
        Register a listener for a specific event type.

        Use "*" to listen to all events.
        """
        self._event_listeners[event_type].append(handler)

    # ── State management ────────────────────────────────────────────

    def reset(self):
        """Clear message history and usage (preserves system prompt)."""
        self._messages = [m for m in self._messages if isinstance(m, SystemMessage)]
        self._usage.reset()

    def get_state(self) -> dict:
        """Return a serializable snapshot of the agent's state."""
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self._messages],
        }

    def load_state(self, state: dict):
        """Restore agent state from a previous get_state() snapshot."""
        self.name = state.get("name", self.name)
        self._messages = [_message_from_dict(m) for m in state.get("messages", [])]

    # ── Properties ──────────────────────────────────────────────────

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    @property
    def total_usage(self) -> Usage:
        return self._usage.total
