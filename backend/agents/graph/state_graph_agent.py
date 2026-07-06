"""
StateGraphAgent — graph with a typed shared state.

Each node receives and returns a shared state dictionary, enabling
nodes to pass data between each other. The state schema can be
enforced via a factory function or dataclass.

This is the most flexible graph pattern, suitable for complex
multi-step workflows (e.g., research → outline → write → review).
"""

from typing import Any, Callable, Generator, Optional

from backend.core.base.tools.tool import Tool
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse
from backend.agents.events.event import AgentEvent, CompletionEvent
from backend.agents.events.stream_event import (
    StreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
)
from backend.agents.graph.graph_agent import GraphAgent, Node, Edge


class StateNode(Node):
    """
    A graph node that operates on a shared state dictionary.

    The node's agent.run() receives the current user_input (which
    can be omitted or set to "" for state-driven workflows), and
    the agent is expected to return its result in the standard way.
    The state is updated externally via the StateGraphAgent.
    """

    pass


class StateGraphAgent(GraphAgent):
    """
    Graph with shared state that flows through nodes.

    Each node receives the current state and can read/write to it.
    State is accessible via self.state inside the agent's run method
    (agents should be designed to accept state-aware inputs).

    Example:
        def initializer():
            return {"topic": "", "outline": "", "content": "", "review": ""}

        graph = StateGraphAgent(
            provider=llm,
            nodes=[StateNode("research", ResearchAgent(...)), ...],
            edges=[Edge("research", "write")],
            entry_point="research",
            state_factory=initializer,
        )
        response = graph.run("Write about AGI safety")
        print(graph.state)  # {"topic": "AGI safety", "outline": "...", ...}
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        nodes: list[StateNode],
        edges: list[Edge],
        entry_point: str,
        terminal_nodes: Optional[list[str]] = None,
        tools: Optional[list[Tool]] = None,
        system_prompt: Optional[str] = None,
        callbacks: Optional[CallbackManager] = None,
        name: Optional[str] = None,
        state_factory: Optional[Callable[[], dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(
            provider=provider,
            nodes=nodes,
            edges=edges,
            entry_point=entry_point,
            terminal_nodes=terminal_nodes,
            tools=tools,
            system_prompt=system_prompt,
            callbacks=callbacks,
            name=name,
            **kwargs,
        )
        self._state_factory = state_factory or (lambda: {})
        self._state: dict[str, Any] = {}

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    def _execute_node(self, node: Node, user_input: str, **kwargs) -> str:
        """Execute a node, passing state-aware context."""
        result = super()._execute_node(node, user_input, **kwargs)
        self._state[node.name] = result
        return result

    def run(self, user_input: str, **kwargs) -> LLMResponse:
        self._state = self._state_factory()
        self._state["user_input"] = user_input
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        response = super().run(user_input, **kwargs)

        self._state["final_output"] = response.message.content
        self.emit(CompletionEvent(response.message.content, response.usage, self.name))

        return response

    def run_stream(
        self, user_input: str, **kwargs
    ) -> Generator[StreamEvent, None, LLMResponse]:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))
        try:
            response = self.run(user_input, **kwargs)
            yield DoneStreamEvent(response.usage)
            return response
        except Exception as e:
            yield ErrorStreamEvent(e)
            self.emit(AgentEvent("error", {"message": str(e)}, self.name))
            raise

    def reset(self):
        super().reset()
        self._state = {}
