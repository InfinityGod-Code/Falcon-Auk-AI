from abc import ABC, abstractmethod
import inspect
from typing import Any, Callable


class Tool(ABC):
    def __init__(self, name, description, func):
        self.name: str = name
        self.description: str = description
        self.func: Callable[..., Any] = func
        self.signature = inspect.signature(self.func)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    async def async_run(self, *args, **kwargs):
        """
        Async execution wrapper.  Default implementation calls ``run()``
        synchronously in an executor thread.  Subclasses like ``MCPTool``
        override this with a native async path.
        """
        import asyncio

        return await asyncio.to_thread(self.run, *args, **kwargs)

    @abstractmethod
    def to_model_specific(self, model_provider):
        pass
