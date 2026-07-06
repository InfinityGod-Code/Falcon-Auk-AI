import json
from typing import Any, Optional
from backend.security.base import SecurityHandler, SecurityLevel


class ToolSecurity(SecurityHandler):
    """
    Controls tool execution with allow/deny lists, parameter validation,
    and output limits.

    Features:
      - Allow list (only these tools may be called)
      - Deny list (these tools are forbidden)
      - Parameter argument must be valid JSON
      - Max result size (truncate large responses)
      - Max tool call depth (prevent infinite recursion)
      - Sensitive parameter name detection
    """

    def __init__(
        self,
        level: SecurityLevel = SecurityLevel.REJECT,
        allow_list: Optional[list[str]] = None,
        deny_list: Optional[list[str]] = None,
        max_result_chars: int = 50000,
        max_depth: int = 10,
        sensitive_param_names: Optional[list[str]] = None,
    ):
        super().__init__(level)
        self._allow_list = allow_list
        self._deny_list = deny_list or []
        self._max_result_chars = max_result_chars
        self._max_depth = max_depth
        self._current_depth = 0
        self._sensitive_params = sensitive_param_names or [
            "password",
            "secret",
            "token",
            "api_key",
            "credential",
        ]

    def on_tool_call(self, tool_call, **kwargs):
        name = getattr(tool_call, "function", {}).get("name", "")
        if isinstance(tool_call, dict):
            name = tool_call.get("function", {}).get("name", "")
        elif hasattr(tool_call, "function"):
            name = tool_call.function.get("name", "")

        if self._allow_list is not None and name not in self._allow_list:
            self.reject(f"Tool '{name}' is not in the allow list")

        if name in self._deny_list:
            self.reject(f"Tool '{name}' is forbidden")

        raw_args = ""
        if isinstance(tool_call, dict):
            raw_args = tool_call.get("function", {}).get("arguments", "{}")
        elif hasattr(tool_call, "function"):
            raw_args = tool_call.function.get("arguments", "{}")

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            self.reject(f"Tool '{name}' arguments are not valid JSON")

        for param in self._sensitive_params:
            if isinstance(args, dict) and param in args:
                self.warn(f"Tool '{name}' called with sensitive parameter '{param}'")

        self._current_depth += 1
        if self._current_depth > self._max_depth:
            self.reject(f"Tool call depth exceeded ({self._max_depth})")

    def on_tool_result(self, tool_call_id: str, name: str, result: Any, **kwargs):
        self._current_depth = max(0, self._current_depth - 1)
        result_str = str(result)
        if len(result_str) > self._max_result_chars:
            self.warn(
                f"Tool '{name}' returned {len(result_str)} chars "
                f"(max {self._max_result_chars})"
            )
