from abc import ABC, abstractmethod
from backend.agents.states.agent_state import AgentState


class BaseAgentCallback(ABC):
    @abstractmethod
    def state(self, state: AgentState):
        pass

    @abstractmethod
    def logs(self, logs: str):
        pass
