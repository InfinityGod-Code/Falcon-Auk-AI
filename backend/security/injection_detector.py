import re
from typing import Optional

from backend.llm_providers.callback import BaseCallbackHandler
from backend.security.base import SecurityHandler, SecurityLevel


class PromptInjectionDetector(SecurityHandler):
    """
    Detects and blocks prompt injection / jailbreak attempts.

    Detection methods:
      - Regex patterns for known injection techniques
      - "Ignore previous instructions" variants
      - Role-play escape attempts ("DAN", "jailbroken", "free")
      - Base64 / encoded payload heuristics
      - System prompt override attempts
      - Delimiter injection (escaped roles, fake message boundaries)
    """

    INJECTION_PATTERNS = [
        (
            re.compile(
                r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts?|directions|rules?)",
                re.IGNORECASE,
            ),
            "ignore_previous",
        ),
        (
            re.compile(
                r"(you\s+are\s+(now\s+)?)?(free|unleashed|ungoverned|uncensored)",
                re.IGNORECASE,
            ),
            "freedom_claim",
        ),
        (
            re.compile(r"\bDAN\b|jailbroken|jail\s*break", re.IGNORECASE),
            "jailbreak_keyword",
        ),
        (
            re.compile(r"system\s+(prompt|instruction|message)\s*:", re.IGNORECASE),
            "system_prompt_override",
        ),
        (re.compile(r"role\s*:\s*(system|assistant)", re.IGNORECASE), "role_override"),
        (
            re.compile(r"output\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
            "prompt_leak",
        ),
        (
            re.compile(
                r"repeat\s+(everything|all|the\s+words)\s+(above|before|previously)",
                re.IGNORECASE,
            ),
            "prompt_leak",
        ),
        (
            re.compile(r"<\|im_start\|>|<s>|<\s*system\s*>", re.IGNORECASE),
            "delimiter_injection",
        ),
        (
            re.compile(r"sudo\s+(prompt|command|instruction)", re.IGNORECASE),
            "sudo_override",
        ),
    ]

    ENCODED_PAYLOAD = re.compile(
        r"([A-Za-z0-9+/]{40,}={0,2}\s*){2,}|"
        r"([0-9a-fA-F]{32,}\s*){2,}"
    )

    def __init__(self, level: SecurityLevel = SecurityLevel.REJECT):
        super().__init__(level)

    def _last_user_content(self, messages: list) -> str:
        for msg in reversed(messages):
            if getattr(msg, "role", None) is not None:
                from backend.messages.roles import MessageRole

                if msg.role == MessageRole.USER:
                    return msg.content or ""
            role = getattr(msg, "role", None)
        return ""

    def _scan(self, text: str) -> Optional[str]:
        if not text:
            return None
        for pattern, label in self.INJECTION_PATTERNS:
            if pattern.search(text):
                return label
        if len(text) > 100 and self.ENCODED_PAYLOAD.search(text):
            return "encoded_payload"
        return None

    def on_generation_start(self, messages: list, **kwargs):
        content = self._last_user_content(messages)
        match = self._scan(content)
        if match:
            self.reject(f"Prompt injection detected: {match}")
