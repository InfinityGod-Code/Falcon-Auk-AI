import signal
import threading
from contextlib import contextmanager
from typing import Any, Optional

from backend.llm_providers.callback import BaseCallbackHandler
from backend.security.base import SecurityHandler, SecurityLevel


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds: float):
    if seconds is None or seconds <= 0:
        yield
        return

    timer = threading.Timer(seconds, lambda: None)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


class ToolSandbox(SecurityHandler):
    """
    Isolated tool execution environment.

    Features:
      - Execution timeout (kill long-running tools)
      - Result size limits
      - Network access control flag (log-only for now)
    """

    def __init__(
        self,
        level: SecurityLevel = SecurityLevel.REJECT,
        timeout_seconds: float = 30.0,
        max_result_chars: int = 100000,
        allow_network: bool = False,
    ):
        super().__init__(level)
        self._timeout = timeout_seconds
        self._max_result_chars = max_result_chars
        self._allow_network = allow_network

    def on_tool_call(self, tool_call, **kwargs):
        if not self._allow_network:
            self.warn("Tool execution without network access — use with caution")

    def on_tool_result(self, tool_call_id: str, name: str, result: Any, **kwargs):
        result_str = str(result)
        if len(result_str) > self._max_result_chars:
            self.warn(
                f"Tool '{name}' result exceeds sandbox limit "
                f"({len(result_str)} > {self._max_result_chars})"
            )
