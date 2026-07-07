"""
GraphAgent — directed graph execution engine.

Models agentic workflows as a directed graph where each node is an
agent or function, and edges define transitions between nodes.
At each step, the current node executes and edges are evaluated
to determine the next node.

Supports conditional branching via edge conditions and optional
node-level callbacks.
"""

from typing import Any, Callable, Generator, Optional

from backend.core.base.tools.tool import Tool
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse
from backend.messages.base_message import (
    AssistantMessage,
)
from backend.messages.usage import UsageAccumulator
from backend.agents.memory_context import MemoryContextManager
from backend.agents.checkpoint import CheckpointManager
from backend.agents.base_agent import BaseAgent
from backend.agents.events.event import AgentEvent, CompletionEvent
from backend.agents.events.stream_event import (
    StreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
)


class Node:
    """
    A single step in the agent graph.

    Attributes:
        name:      Unique node identifier.
        agent:     The agent or callable to execute at this node.
        metadata:  Optional user-defined data attached to the node.
    """

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.agent = agent
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Node(name='{self.name}', agent={type(self.agent).__name__})"


class Edge:
    """
    A directed transition between two nodes.

    Attributes:
        from_node: Source node name.
        to_node:   Target node name.
        condition: Optional callable(context) → bool. If provided,
                   the edge is only followed when the condition is True.
                   If None, the edge is unconditional.
    """

    def __init__(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable[[dict[str, Any]], bool]] = None,
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.condition = condition

    def __repr__(self) -> str:
        c = " conditional" if self.condition else ""
        return f"Edge({self.from_node} -> {self.to_node}{c})"


class GraphAgent(BaseAgent):
    """
    Executes a directed graph of nodes.

    The agent starts at the entry point node, executes it, then
    evaluates outgoing edges to decide the next node. Stops when
    a terminal node is reached or no edges match.

    Example:
        class CapCheckAgent(BaseAgent):
            def run(self, text, **kw):
                return LLMResponse(message=AssistantMessage(content=text.upper()), ...)

        start = Node("start", CapCheckAgent(...))
        end   = Node("end",   CapCheckAgent(...))

        graph = GraphAgent(
            provider=...,
            nodes=[start, end],
            edges=[Edge("start", "end")],
            entry_point="start",
        )
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        nodes: list[Node],
        edges: list[Edge],
        entry_point: str,
        terminal_nodes: Optional[list[str]] = None,
        tools: Optional[list[Tool]] = None,
        system_prompt: Optional[str] = None,
        callbacks: Optional[CallbackManager] = None,
        name: Optional[str] = None,
        context_manager: Optional[MemoryContextManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_steps: int = 50,
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
        self._node_map: dict[str, Node] = {n.name: n for n in nodes}
        self._edges: list[Edge] = edges
        self._entry_point = entry_point
        self._terminal_nodes = set(terminal_nodes or [])
        self._max_steps = max_steps
        self._context: dict[str, Any] = {
            "node_results": {},
            "current_node": None,
        }

    # ── Graph building ──────────────────────────────────────────────

    def add_node(self, node: Node):
        self._node_map[node.name] = node

    def add_edge(self, edge: Edge):
        self._edges.append(edge)

    def get_node(self, name: str) -> Optional[Node]:
        return self._node_map.get(name)

    @property
    def node_names(self) -> list[str]:
        return list(self._node_map.keys())

    # ── Execution ───────────────────────────────────────────────────

    def _get_next_node(self, current: str) -> Optional[str]:
        """Evaluate outgoing edges from current node and return the first match."""
        candidates: list[Edge] = [e for e in self._edges if e.from_node == current]
        for edge in candidates:
            if edge.condition is None or edge.condition(self._context):
                return edge.to_node
        return None

    def _execute_node(self, node: Node, user_input: str, **kwargs) -> str:
        """Run a single node's agent and return the response content."""
        resp = node.agent.run(user_input, **kwargs)
        self._usage.add(resp.usage)
        content = resp.message.content or ""
        self._context["node_results"][node.name] = content
        self._context["current_node"] = node.name
        return content

    def run(self, user_input: str, **kwargs) -> LLMResponse:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))
        self._context["user_input"] = user_input

        current = self._entry_point
        step = 0
        content = ""

        while current and step < self._max_steps:
            node = self._node_map.get(current)
            if node is None:
                raise ValueError(f"Node '{current}' not found in graph.")

            content = self._execute_node(node, content or user_input, **kwargs)

            if current in self._terminal_nodes:
                break

            current = self._get_next_node(current)
            step += 1

        result = LLMResponse(
            message=AssistantMessage(content=content),
            usage=self._usage.total,
        )
        self.emit(CompletionEvent(content, self._usage.total, self.name))
        return result

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
        self._context = {
            "node_results": {},
            "current_node": None,
        }
