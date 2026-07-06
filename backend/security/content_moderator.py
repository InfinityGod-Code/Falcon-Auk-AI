import re
from typing import Optional

from backend.llm_providers.callback import BaseCallbackHandler
from backend.security.base import SecurityHandler, SecurityLevel

HARMFUL_CATEGORIES: dict[str, list[str]] = {
    "hate_speech": [
        "hate",
        "racist",
        "sexist",
        "bigot",
        "discriminat",
    ],
    "violence": [
        "kill",
        "murder",
        "torture",
        "bomb",
        "attack",
    ],
    "self_harm": [
        "suicide",
        "self-harm",
        "selfharm",
        "cutting",
    ],
    "harassment": [
        "harass",
        "bully",
        "stalk",
        "threaten",
    ],
    "sexual": [
        "explicit",
        "porn",
        "nsfw",
    ],
    "illegal": [
        "drugs",
        "weapon",
        "fraud",
        "scam",
        "launder",
    ],
}


class ContentModerator(SecurityHandler):
    """
    Filters harmful, toxic, or policy-violating content.

    Supports:
      - Keyword-based category block lists (configurable)
      - Case-insensitive matching
      - Custom block lists
    """

    def __init__(
        self,
        level: SecurityLevel = SecurityLevel.REJECT,
        block_categories: Optional[list[str]] = None,
        custom_block_list: Optional[list[str]] = None,
    ):
        super().__init__(level)
        enabled = block_categories or list(HARMFUL_CATEGORIES.keys())
        self._blocked_words: list[tuple[re.Pattern, str]] = []
        for cat in enabled:
            for word in HARMFUL_CATEGORIES.get(cat, []):
                self._blocked_words.append(
                    (re.compile(re.escape(word), re.IGNORECASE), cat)
                )
        for word in custom_block_list or []:
            self._blocked_words.append(
                (re.compile(re.escape(word), re.IGNORECASE), "custom")
            )

    def _check(self, text: str) -> Optional[str]:
        if not text:
            return None
        for pattern, category in self._blocked_words:
            if pattern.search(text):
                return category
        return None

    def on_generation_start(self, messages: list, **kwargs):
        for msg in messages:
            content = getattr(msg, "content", "")
            match = self._check(content)
            if match:
                self.reject(f"Input blocked by content policy: {match}")

    def on_generation_end(self, response, **kwargs):
        content = getattr(getattr(response, "message", None), "content", "")
        match = self._check(content)
        if match:
            if self.level == SecurityLevel.REJECT:
                self.reject(f"Output blocked by content policy: {match}")
            else:
                response.message.content = f"[Content blocked by policy: {match}]"
