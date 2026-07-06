from abc import ABC
from typing import Any, Optional


class BaseCallbackHandler(ABC):
    def on_generation_start(
        self, messages: Any, tools: Any, model: str, **kwargs
    ) -> None:
        pass

    def on_generation_end(self, response: Any, usage: Any, **kwargs) -> None:
        pass

    def on_stream_chunk(self, chunk: Any, **kwargs) -> None:
        pass

    def on_tool_call(self, tool_call: Any, **kwargs) -> None:
        pass

    def on_tool_result(
        self, tool_call_id: str, name: str, result: Any, **kwargs
    ) -> None:
        pass

    def on_error(self, error: Exception, **kwargs) -> None:
        pass

    def on_retry(self, attempt: int, error: Exception, **kwargs) -> None:
        pass


class CallbackManager:
    def __init__(self, handlers: Optional[list[BaseCallbackHandler]] = None):
        self._handlers: list[BaseCallbackHandler] = handlers or []

    def add_handler(self, handler: BaseCallbackHandler):
        self._handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler):
        self._handlers.remove(handler)

    def on_generation_start(self, **kwargs):
        for h in self._handlers:
            h.on_generation_start(**kwargs)

    def on_generation_end(self, **kwargs):
        for h in self._handlers:
            h.on_generation_end(**kwargs)

    def on_stream_chunk(self, **kwargs):
        for h in self._handlers:
            h.on_stream_chunk(**kwargs)

    def on_tool_call(self, **kwargs):
        for h in self._handlers:
            h.on_tool_call(**kwargs)

    def on_tool_result(self, **kwargs):
        for h in self._handlers:
            h.on_tool_result(**kwargs)

    def on_error(self, **kwargs):
        for h in self._handlers:
            h.on_error(**kwargs)

    def on_retry(self, **kwargs):
        for h in self._handlers:
            h.on_retry(**kwargs)
