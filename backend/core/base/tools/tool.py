from abc import ABC, abstractmethod
import inspect
from typing import Callable, Any


class Tool(ABC):
    def __init__(self, name, description, func):
        self.name: str = name
        self.description: str = description
        self.func: Callable[..., Any] = func
        self.signature = inspect.signature(self.func)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def to_model_specific(self, model_provider):
        pass
