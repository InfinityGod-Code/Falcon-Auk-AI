"""
WorkflowAgent — DAG-based workflow with parallel execution.

Models agentic workflows as a Directed Acyclic Graph (DAG) where
nodes can run in parallel when their dependencies are satisfied.
Uses topological ordering to schedule execution and joins parallel
branches at synchronization points.

Suitable for tasks like: research → (write + illustrate) → merge → review
"""

from typing import Any, Callable, Generator, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque

from backend.core.base.tools.tool import Tool
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.response import LLMResponse
from backend.messages.base_message import AssistantMessage
from backend.agents.base_agent import BaseAgent
from backend.agents.events.event import AgentEvent, CompletionEvent
from backend.agents.events.stream_event import (
    StreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
)
from backend.agents.graph.graph_agent import Node, Edge, GraphAgent


class WorkflowAgent(GraphAgent):
    """
    DAG-based agent that executes nodes in topological order,
    running independent branches in parallel.

    Nodes without dependencies run concurrently via a thread pool.
    Nodes with all dependencies satisfied proceed; others wait.

    Example:
        workflow = WorkflowAgent(
            provider=llm,
            nodes=[
                Node("research", ResearchAgent(...)),
                Node("write",    WriteAgent(...)),
                Node("illustrate", IllustrateAgent(...)),
                Node("merge",    MergeAgent(...)),
                Node("review",   ReviewAgent(...)),
            ],
            edges=[
                Edge("research", "write"),
                Edge("research", "illustrate"),
                Edge("write", "merge"),
                Edge("illustrate", "merge"),
                Edge("merge", "review"),
            ],
            entry_point="research",
            terminal_nodes=["review"],
        )
        response = workflow.run("Create a blog post about AI")
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
        max_workers: int = 4,
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
        self._max_workers = max_workers

    def _build_dag(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """
        Build adjacency list and in-degree map from the edge list.
        Returns (children, in_degree).
        """
        children: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = defaultdict(int)

        for name in self._node_map:
            in_degree[name] = 0

        for edge in self._edges:
            if edge.from_node in self._node_map and edge.to_node in self._node_map:
                # Respect conditions: skip conditioned edges for automatic scheduling
                if edge.condition is None:
                    children[edge.from_node].append(edge.to_node)
                    in_degree[edge.to_node] += 1

        return children, in_degree

    def _execute_workflow(self, user_input: str, **kwargs) -> str:
        """
        Execute the DAG using topological ordering with parallel fan-out.
        """
        children, in_degree = self._build_dag()

        ready: deque[str] = deque()
        for name, degree in in_degree.items():
            if degree == 0:
                ready.append(name)

        results: dict[str, str] = {}
        node_inputs: dict[str, str] = {}
        node_inputs[self._entry_point] = user_input

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:

            def _run_node(name: str) -> tuple[str, str]:
                node = self._node_map[name]
                inp = node_inputs.get(name, user_input)
                content = self._execute_node(node, inp, **kwargs)
                return name, content

            while ready:
                batch = list(ready)
                ready.clear()

                futures = {pool.submit(_run_node, name): name for name in batch}

                for future in as_completed(futures):
                    name, content = future.result()
                    results[name] = content

                    for child in children.get(name, []):
                        in_degree[child] -= 1
                        if in_degree[child] == 0:
                            node_inputs[child] = content
                            ready.append(child)

        terminal = self._terminal_nodes & results.keys()
        if terminal:
            last = max(terminal, key=lambda n: list(results.keys()).index(n))
            return results[last]

        last_node = max(results.keys(), key=lambda n: list(results.keys()).index(n))
        return results[last_node]

    def run(self, user_input: str, **kwargs) -> LLMResponse:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))

        content = self._execute_workflow(user_input, **kwargs)

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
