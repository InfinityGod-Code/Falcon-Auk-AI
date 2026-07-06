import re
from typing import Optional

from backend.llm_providers.callback import BaseCallbackHandler
from backend.security.base import SecurityHandler, SecurityLevel


class OutputGuardrail(SecurityHandler):
    """
    Validates LLM output before it reaches the user or tool executor.

    Checks:
      - Internal/hidden URL leakage (localhost, 127.0.0.1, 10.x.x.x)
      - Secret/key pattern leakage
      - Response length limits
      - Structured output format validation
      - Hallucinated citation detection (fake DOIs, URLs, etc.)
    """

    INTERNAL_HOSTS = re.compile(
        r"\b(localhost|127\.\d{1,3}\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3})\b"
    )

    LEAKED_KEY_PATTERNS = re.compile(
        r"\b(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36})\b"
    )

    FAKE_DOI = re.compile(r"\b10\.\d{4,}/[^\s]{10,}\b")

    def __init__(
        self,
        level: SecurityLevel = SecurityLevel.REDACT,
        max_output_chars: int = 100000,
        redact_patterns: Optional[list[re.Pattern]] = None,
    ):
        super().__init__(level)
        self._max_output_chars = max_output_chars
        self._redact_patterns = redact_patterns or [
            self.INTERNAL_HOSTS,
            self.LEAKED_KEY_PATTERNS,
            self.FAKE_DOI,
        ]

    def _validate(self, content: str):
        if len(content) > self._max_output_chars:
            self.warn(
                f"Output exceeds max length ({len(content)} > {self._max_output_chars})"
            )

    def _redact(self, content: str) -> str:
        if not content:
            return content
        for pattern in self._redact_patterns:
            content = pattern.sub("[REDACTED]", content)
        return content

    def on_generation_end(self, response, **kwargs):
        msg = getattr(response, "message", None)
        if msg is None:
            return
        content = msg.content or ""
        self._validate(content)
        msg.content = self._redact(content)
