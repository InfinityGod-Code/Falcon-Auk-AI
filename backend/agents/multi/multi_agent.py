"""
MultiAgent — composite base for multi-agent systems.

Uses the Composite pattern: treats a group of named sub-agents
uniformly. Subclasses (SupervisorAgent, SwarmAgent) define
specific orchestration strategies.

Sub-agents are registered by name and can be any BaseAgent.
"""

from abc import ABC
from typing import Any, AsyncGenerator, Optional

from backend.core.base.tools.tool import Tool
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse
from backend.messages.base_message import BaseMessage
from backend.messages.usage import UsageAccumulator
from backend.agents.memory_context import MemoryContextManager
from backend.agents.checkpoint import CheckpointManager
from backend.agents.base_agent import BaseAgent


class MultiAgent(BaseAgent, ABC):
    """
    Composite container for multiple named sub-agents.

    Provides agent registry (add, get, remove, list) and delegates
    run() / run_stream() to subclasses via abstract methods.

    Sub-agents do NOT share the parent's message history — each
    keeps its own internal state.
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
        agents: Optional[dict[str, BaseAgent]] = None,
        usage_accumulator: Optional[UsageAccumulator] = None,
    ):
        super().__init__(
            provider,
            tools,
            system_prompt,
            callbacks,
            name,
            context_manager=context_manager,
            checkpoint_manager=checkpoint_manager,
            usage_accumulator=usage_accumulator,
        )
        self._agents: dict[str, BaseAgent] = agents or {}

    # ── Sub-agent registry ──────────────────────────────────────────

    def add_agent(self, name: str, agent: BaseAgent):
        """Register a sub-agent with a unique name."""
        self._agents[name] = agent

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Retrieve a registered sub-agent by name."""
        return self._agents.get(name)

    def remove_agent(self, name: str):
        """Unregister a sub-agent."""
        self._agents.pop(name, None)

    @property
    def agent_names(self) -> list[str]:
        """List all registered sub-agent names."""
        return list(self._agents.keys())

    def list_agents(self) -> dict[str, BaseAgent]:
        """Return the full agent registry."""
        return dict(self._agents)

    # ── Execution (delegated to subclasses) ─────────────────────────

    async def run(self, user_input: str, **kwargs) -> LLMResponse:
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")

    async def run_stream(self, user_input: str, **kwargs) -> AsyncGenerator[Any, None]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run_stream()"
        )
