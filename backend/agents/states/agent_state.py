"""Agent states are the ones that keeps the current state of the Agent in the [AgentLifeCycle]
Agent State are managed here with the help of dataclasses in the Python with Frozen= True
One thing that I have taken care of the nested state inside the Agent are also immutalble
Therefore you may find the Tuples or FrozenSet instead of normal List and Sets in Python for the
implementation. Whenever we want to change the AgentState we need to use the replace method from
the dataclasses itself.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentState:
    total_token_consumed: int | None = None
    current_tool: str | None = None
    message_length: int | None = None
    retry_count: int | None = None
    tool_name: str | None = None
