import json
import logging
import time
from typing import Any, Optional
from pathlib import Path

from backend.llm_providers.callback import BaseCallbackHandler
from backend.security.base import SecurityHandler, SecurityLevel

logger = logging.getLogger("falcon_auk.security")


class AuditLogger(SecurityHandler):
    """
    Structured audit logging for all security-relevant events.

    Supports multiple outputs:
      - Stdlib logging (default)
      - JSON Lines file
      - In-memory buffer (for programmatic access)
    """

    def __init__(
        self,
        level: SecurityLevel = SecurityLevel.LOG_ONLY,
        output: str = "log",
        log_path: Optional[str] = None,
    ):
        super().__init__(level)
        self._output = output
        self._entries: list[dict[str, Any]] = []
        self._log_file = Path(log_path) if log_path else None

        if output == "file" and self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, event_type: str, data: dict[str, Any]):
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            **data,
        }
        self._entries.append(entry)

        if self._output == "log":
            logger.info("[%s] %s", event_type, json.dumps(data))
        elif self._output == "file" and self._log_file:
            line = json.dumps(entry) + "\n"
            with open(self._log_file, "a") as f:
                f.write(line)

    def on_generation_start(self, messages: list, **kwargs):
        self._log(
            "generation_start",
            {
                "message_count": len(messages),
                "model": kwargs.get("model", ""),
            },
        )

    def on_generation_end(self, response, **kwargs):
        usage = getattr(response, "usage", None)
        self._log(
            "generation_end",
            {
                "total_tokens": usage.total_tokens if usage else 0,
            },
        )

    def on_stream_chunk(self, chunk, **kwargs):
        pass

    def on_tool_call(self, tool_call, **kwargs):
        name = ""
        if hasattr(tool_call, "function"):
            name = tool_call.function.get("name", "")
        elif isinstance(tool_call, dict):
            name = tool_call.get("function", {}).get("name", "")
        self._log("tool_call", {"tool_name": name})

    def on_tool_result(self, tool_call_id: str, name: str, result: Any, **kwargs):
        self._log(
            "tool_result",
            {
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "result_length": len(str(result)),
            },
        )

    def on_error(self, error: Exception, **kwargs):
        self._log(
            "error",
            {
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )

    def on_retry(self, attempt: int, error: Exception, **kwargs):
        self._log(
            "retry",
            {
                "attempt": attempt,
                "error": str(error),
            },
        )

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def clear(self):
        self._entries.clear()
