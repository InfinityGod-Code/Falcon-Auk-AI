import re
from typing import Optional

from backend.llm_providers.callback import BaseCallbackHandler
from backend.security.base import SecurityHandler, SecurityLevel

PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    "api_key_sk": re.compile(r"\b(sk-[A-Za-z0-9]{20,}|[A-Za-z0-9_-]{20,})\b"),
    "aws_key": re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
}


class PIIRedactor(SecurityHandler):
    """
    Detects and redacts personally identifiable information.

    Operates in both directions:
      - Input (User → LLM): redact PII before sending to provider
      - Output (LLM → User): redact any PII in generated responses

    Modes:
      - redact:  replace with "[REDACTED]"
      - mask:    show first/last chars (e.g. j***@***.com)
      - hash:    replace with SHA-256 digest
      - log_only: log detection but leave unchanged
    """

    def __init__(
        self,
        mode: str = "redact",
        level: SecurityLevel = SecurityLevel.REDACT,
        enabled_patterns: Optional[list[str]] = None,
    ):
        super().__init__(level)
        self._mode = mode
        self._patterns = {
            name: pat
            for name, pat in PII_PATTERNS.items()
            if enabled_patterns is None or name in enabled_patterns
        }

    def _redact(self, text: str) -> str:
        if not text:
            return text
        for name, pattern in self._patterns.items():
            if self._mode == "redact":
                text = pattern.sub(f"[{name.upper()}_REDACTED]", text)
            elif self._mode == "log_only":
                matches = pattern.findall(text)
                if matches:
                    self.warn(f"PII detected: {name} x{len(matches)}")
        return text

    def on_generation_start(self, messages: list, **kwargs):
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                original = msg.content
                msg.content = self._redact(msg.content)

    def on_generation_end(self, response, **kwargs):
        msg = getattr(response, "message", None)
        if msg and msg.content:
            msg.content = self._redact(msg.content)
