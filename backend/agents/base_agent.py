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
from typing import Any, AsyncGenerator, Callable, Optional

from backend.core.base.tools.tool import Tool
from backend.messages.base_message import BaseMessage
from backend.messages.usage import Usage, UsageAccumulator
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse
from backend.agents.memory_context import MemoryContextManager
from backend.agents.checkpoint import CheckpointManager
from backend.agents.events.event import AgentEvent, CheckpointCreatedEvent


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
        context_manager: Optional[MemoryContextManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        usage_accumulator: Optional[UsageAccumulator] = None,
    ):
        self.provider = provider
        self.tools = tools
        self.callbacks = callbacks or CallbackManager()
        self.system_prompt = system_prompt
        self.name = name or self.__class__.__name__

        self.context = context_manager or MemoryContextManager(
            system_prompt=system_prompt
        )
        self.checkpoints = checkpoint_manager
        self._usage = usage_accumulator or UsageAccumulator()
        self._event_listeners: dict[str, list[Callable[[AgentEvent], None]]] = (
            defaultdict(list)
        )

    @abstractmethod
    async def run(self, user_input: str, **kwargs) -> LLMResponse:
        """Execute the agent on a single user input and return a response."""
        ...

    @abstractmethod
    async def run_stream(self, user_input: str, **kwargs) -> AsyncGenerator[Any, None]:
        """
        Execute the agent in streaming mode.

        Yields stream events (tokens, tool_calls, etc.)
        and returns the final LLMResponse via StopAsyncIteration.
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
        self.context.clear(keep_system=True)
        self._usage.reset()

    def get_state(self) -> dict:
        """Return a serializable snapshot of the agent's state."""
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.context.get_messages()],
            "context": self.context.to_dict(),
            "usage": {
                "prompt_tokens": self.total_usage.prompt_tokens,
                "completion_tokens": self.total_usage.completion_tokens,
                "total_tokens": self.total_usage.total_tokens,
            },
        }

    def load_state(self, state: dict):
        """Restore agent state from a previous get_state() snapshot."""
        self.name = state.get("name", self.name)
        ctx_data = state.get("context")
        if ctx_data:
            self.context = MemoryContextManager.from_dict(ctx_data)
        else:
            from backend.llm_providers.lifecycle import _message_from_dict

            self.context.replace_messages(
                [_message_from_dict(m) for m in state.get("messages", [])]
            )

    def create_checkpoint(self, metadata: Optional[dict] = None) -> Optional[str]:
        """Create a checkpoint of current agent state."""
        if not self.checkpoints:
            return None
        cp = self.checkpoints.save(
            agent_name=self.name,
            context_data=self.context.to_dict(),
            usage_data={
                "prompt_tokens": self.total_usage.prompt_tokens,
                "completion_tokens": self.total_usage.completion_tokens,
                "total_tokens": self.total_usage.total_tokens,
            },
            provider_type=self.provider.provider.value
            if hasattr(self.provider, "provider")
            else "",
            provider_model=self.provider.model
            if hasattr(self.provider, "model")
            else "",
            metadata=metadata,
        )
        self.emit(
            CheckpointCreatedEvent(checkpoint_id=cp.checkpoint_id, agent=self.name)
        )
        return cp.checkpoint_id

    def restore_from_checkpoint(self, checkpoint_id: str):
        """Restore agent state from a checkpoint."""
        if not self.checkpoints:
            raise RuntimeError("No CheckpointManager configured")
        self.checkpoints.restore(checkpoint_id, self.context, self._usage)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def messages(self) -> list[BaseMessage]:
        return self.context.get_messages()

    @property
    def total_usage(self) -> Usage:
        return self._usage.total
