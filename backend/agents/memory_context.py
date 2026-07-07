"""
ContextManager — single source of truth for agent context.

Manages message history, shared variables, token budget, and
context window strategies (sliding window, token budget trim,
summarization). Decouples context logic from agent execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from backend.messages.base_message import (
    BaseMessage,
    SystemMessage,
)
from backend.llm_providers.lifecycle import _message_from_dict


class MemoryContextStrategy(ABC):
    """
    Strategy for managing context within a token budget.

    Subclasses implement apply() to transform the message list
    before it is sent to the LLM.
    """

    @abstractmethod
    def apply(
        self,
        messages: list[BaseMessage],
        max_tokens: int,
        token_counter: Optional[Callable[[list[BaseMessage]], int]] = None,
    ) -> list[BaseMessage]:
        """
        Return a (possibly trimmed/summarised) copy of messages
        that fits within max_tokens.
        """
        ...


class NoOpStrategy(MemoryContextStrategy):
    """Pass through — no trimming applied."""

    def apply(
        self,
        messages: list[BaseMessage],
        max_tokens: int,
        token_counter: Optional[Callable[[list[BaseMessage]], int]] = None,
    ) -> list[BaseMessage]:
        return list(messages)


class SlidingWindowStrategy(MemoryContextStrategy):
    """
    Keep the last N messages. Drops older messages beyond the window.

    The system prompt (first message) is always preserved.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def apply(
        self,
        messages: list[BaseMessage],
        max_tokens: int,
        token_counter: Optional[Callable[[list[BaseMessage]], int]] = None,
    ) -> list[BaseMessage]:
        if len(messages) <= self.window_size:
            return list(messages)

        system = [m for m in messages if isinstance(m, SystemMessage)]
        others = [m for m in messages if not isinstance(m, SystemMessage)]
        trimmed = (
            others[-(self.window_size - len(system)) :]
            if system
            else others[-self.window_size :]
        )
        return system + trimmed


class TokenBudgetStrategy(MemoryContextStrategy):
    """
    Trim oldest non-system messages until the total estimated
    token count fits within max_tokens.
    """

    def _estimate_tokens(self, messages: list[BaseMessage]) -> int:
        return sum(len(m.content) // 4 for m in messages)

    def apply(
        self,
        messages: list[BaseMessage],
        max_tokens: int,
        token_counter: Optional[Callable[[list[BaseMessage]], int]] = None,
    ) -> list[BaseMessage]:
        counter = token_counter or self._estimate_tokens
        result = list(messages)

        while result and counter(result) > max_tokens:
            # Drop oldest non-system message
            dropped = False
            for i, m in enumerate(result):
                if not isinstance(m, SystemMessage):
                    result.pop(i)
                    dropped = True
                    break
            if not dropped:
                break

        return result


class SummarizationStrategy(MemoryContextStrategy):
    """
    Replace oldest messages with a single summary system message
    when the token budget is exceeded.
    """

    def __init__(self, summary_prompt: str = "Previous conversation summary:"):
        self.summary_prompt = summary_prompt
        self._summary: Optional[str] = None

    def apply(
        self,
        messages: list[BaseMessage],
        max_tokens: int,
        token_counter: Optional[Callable[[list[BaseMessage]], int]] = None,
    ) -> list[BaseMessage]:
        counter = token_counter or (lambda msgs: sum(len(m.content) // 4 for m in msgs))

        if not messages or counter(messages) <= max_tokens:
            return list(messages)

        system = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        # Build summary from first half of non-system messages
        mid = len(non_system) // 2
        to_summarize = non_system[:mid]
        rest = non_system[mid:]

        summary_text = " ".join(m.content[:200] for m in to_summarize if m.content)
        self._summary = f"{self.summary_prompt}\n{summary_text}"
        summary_msg = SystemMessage(content=self._summary)

        return system + [summary_msg] + rest

    @property
    def summary(self) -> Optional[str]:
        return self._summary


class MemoryContextManager:
    """
    Owns all agent context: messages, variables, metadata, and token budget.

    Acts as the single source of truth that both BaseAgent and
    LLMLifecycle delegate to — eliminating duplicate message tracking.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_tokens: int = 128000,
        strategy: Optional[MemoryContextStrategy] = None,
        token_counter: Optional[Callable[[list[BaseMessage]], int]] = None,
    ):
        self._messages: list[BaseMessage] = []
        self._variables: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._max_tokens = max_tokens
        self._strategy = strategy or NoOpStrategy()
        self._token_counter = token_counter

        if system_prompt:
            self._messages.append(SystemMessage(content=system_prompt))

    def add_message(self, message: BaseMessage):
        self._messages.append(message)

    def add_messages(self, messages: list[BaseMessage]):
        self._messages.extend(messages)

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def get_context(self) -> list[BaseMessage]:
        """
        Return the strategy-applied message list (trimmed/summarised)
        for sending to the LLM.
        """
        return self._strategy.apply(
            self._messages, self._max_tokens, self._token_counter
        )

    def clear(self, keep_system: bool = True):
        if keep_system:
            self._messages = [m for m in self._messages if isinstance(m, SystemMessage)]
        else:
            self._messages = []

    def replace_messages(self, messages: list[BaseMessage]):
        self._messages = list(messages)

    # ── Variables (cross-agent shared state) ────────────────────────

    def set(self, key: str, value: Any):
        self._variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._variables.get(key, default)

    @property
    def variables(self) -> dict[str, Any]:
        return dict(self._variables)

    # ── Token awareness ─────────────────────────────────────────────

    def count_tokens(self) -> int:
        counter = self._token_counter or (
            lambda msgs: sum(len(m.content) // 4 for m in msgs)
        )
        return counter(self._messages)

    def remaining_tokens(self) -> int:
        return self._max_tokens - self.count_tokens()

    def is_within_budget(self) -> bool:
        return self.count_tokens() <= self._max_tokens

    def trim(self) -> list[BaseMessage]:
        return self._strategy.apply(
            self._messages, self._max_tokens, self._token_counter
        )

    # ── Serialization ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "messages": [m.to_dict() for m in self._messages],
            "variables": dict(self._variables),
            "metadata": dict(self._metadata),
            "max_tokens": self._max_tokens,
            "strategy": type(self._strategy).__name__,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryContextManager":
        cm = cls(
            max_tokens=data.get("max_tokens", 128000),
        )
        cm._messages = [_message_from_dict(m) for m in data.get("messages", [])]
        cm._variables = dict(data.get("variables", {}))
        cm._metadata = dict(data.get("metadata", {}))
        return cm
