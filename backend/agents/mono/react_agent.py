"""
ReActAgent — Reasoning + Acting loop (ReAct pattern).

The agent iterates through Thought → Action → Observation cycles
using a text-based prompt pattern. Unlike ToolCallingAgent which
relies on native function calling, ReActAgent instructs the LLM
via the system prompt to output structured thought/action/observation
blocks and parses them to decide the next step.

Useful for models or scenarios where native tool calling is
unavailable or undesired.
"""

import re
from typing import Any, Callable, Generator, Optional

from backend.core.base.tools.tool import Tool
from backend.llm_providers.base import BaseLLMProvider
from backend.llm_providers.callback import CallbackManager
from backend.llm_providers.lifecycle import LLMLifecycle
from backend.llm_providers.response import LLMResponse
from backend.messages.base_message import SystemMessage, UserMessage, AssistantMessage
from backend.agents.context import ContextManager
from backend.agents.checkpoint import CheckpointManager
from backend.agents.base_agent import BaseAgent
from backend.agents.events.event import (
    AgentEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
    CompletionEvent,
)
from backend.agents.events.stream_event import (
    StreamEvent,
    TokenStreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
)

_REACT_SYSTEM_PROMPT = """You are a ReAct agent. You reason step-by-step and call tools to gather information.

You MUST respond with EXACTLY ONE of these formats per turn:

1. To use a tool:
THOUGHT: <your reasoning>
ACTION: <tool_name>
ACTION_INPUT: <JSON arguments>

2. To give the final answer:
THOUGHT: <your reasoning>
FINAL_ANSWER: <your answer to the user>
"""


class ReActAgent(BaseAgent):
    """
    ReAct agent that iterates through Thought → Action → Observation loops.

    Parses the LLM's text output for ACTION/Action_INPUT blocks,
    executes the corresponding tool, feeds the result back as an
    Observation, and repeats until FINAL_ANSWER appears.

    Example:
        agent = ReActAgent(provider=..., tools=[weather_tool])
        response = agent.run("What is the weather in Paris?")
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Optional[list[Tool]] = None,
        system_prompt: Optional[str] = None,
        callbacks: Optional[CallbackManager] = None,
        name: Optional[str] = None,
        context_manager: Optional[ContextManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        tool_executor: Optional[Callable[[str, str], Any]] = None,
        max_steps: int = 10,
    ):
        super().__init__(
            provider,
            tools,
            system_prompt,
            callbacks,
            name,
            context_manager=context_manager,
            checkpoint_manager=checkpoint_manager,
        )
        self._tool_executor = tool_executor
        self._max_steps = max_steps

        prompt = system_prompt or _REACT_SYSTEM_PROMPT
        tool_descriptions = "\n".join(
            f"  - {t.name}: {t.description}" for t in self.tools
        )
        if tool_descriptions:
            prompt += f"\n\nAvailable tools:\n{tool_descriptions}"

        self._lifecycle = LLMLifecycle(
            provider=provider,
            tools=None,
        )
        self._lifecycle.add_message(SystemMessage(content=prompt))

    def _parse_action(self, text: str) -> Optional[tuple[str, str]]:
        """Parse ACTION / ACTION_INPUT blocks from LLM output."""
        action_match = re.search(r"ACTION:\s*(\w+)", text)
        input_match = re.search(
            r"ACTION_INPUT:\s*(\{.*\}|`[^`]+`|\S+)", text, re.DOTALL
        )
        if action_match:
            name = action_match.group(1).strip()
            raw = input_match.group(1).strip() if input_match else "{}"
            raw = raw.strip("`")
            return name, raw
        return None

    def _has_final_answer(self, text: str) -> Optional[str]:
        """Extract FINAL_ANSWER if present."""
        match = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _find_tool(self, name: str) -> Optional[Tool]:
        for t in self.tools:
            if t.name == name or t.name.replace("_", "") == name.replace("_", ""):
                return t
        return None

    def run(self, user_input: str, **kwargs) -> LLMResponse:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))
        self._lifecycle.add_message(UserMessage(content=user_input))

        for step in range(self._max_steps):
            response = self._lifecycle.provider.generate(
                messages=self._lifecycle.messages,
                **kwargs,
            )
            self._lifecycle.messages.append(response.message)
            self._usage.add(response.usage)

            thought = response.message.content or ""
            self.emit(ThoughtEvent(thought, self.name))

            final_answer = self._has_final_answer(thought)
            if final_answer:
                result = AssistantMessage(content=final_answer)
                self._lifecycle.messages.append(result)
                self._messages.append(result)
                self.emit(CompletionEvent(final_answer, response.usage, self.name))
                return LLMResponse(message=result, usage=response.usage)

            parsed = self._parse_action(thought)
            if not parsed:
                self._lifecycle.add_message(
                    AssistantMessage(
                        content="I need to use a tool. Please provide ACTION and ACTION_INPUT."
                    )
                )
                continue

            tool_name, tool_input = parsed
            tool = self._find_tool(tool_name)
            if tool is None or self._tool_executor is None:
                obs = f"Tool '{tool_name}' not found. Available: {[t.name for t in self.tools]}"
            else:
                self.emit(
                    ToolCallEvent(
                        type(
                            "tc",
                            (),
                            {
                                "id": "",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_input,
                                },
                            },
                        )(),
                        self.name,
                    )
                )
                result = self._tool_executor(tool_name, tool_input)
                obs = str(result)
                self.emit(ToolResultEvent("", tool_name, result, self.name))

            from backend.messages.base_message import ToolMessage as TMsg

            self._lifecycle.add_message(AssistantMessage(content=f"OBSERVATION: {obs}"))

        return self._lifecycle.run("Continue and provide your FINAL_ANSWER.")

    def run_stream(
        self, user_input: str, **kwargs
    ) -> Generator[StreamEvent, None, LLMResponse]:
        self.emit(AgentEvent("run_start", {"input": user_input}, self.name))
        self._lifecycle.add_message(UserMessage(content=user_input))

        try:
            response = self.run(user_input, **kwargs)
            yield DoneStreamEvent(response.usage)
            return response
        except Exception as e:
            yield ErrorStreamEvent(e)
            self.emit(AgentEvent("error", {"message": str(e)}, self.name))
            raise
