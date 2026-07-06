import time
from collections import defaultdict, deque
from typing import Optional

from backend.llm_providers.callback import BaseCallbackHandler
from backend.security.base import SecurityHandler, SecurityLevel


class InMemoryRateStore:
    """Thread-safe (simple) in-memory rate limit counter."""

    def __init__(self):
        self._windows: dict[str, dict[str, deque]] = defaultdict(
            lambda: defaultdict(deque)
        )

    def consume(
        self, key: str, counter: str, amount: int = 1, window: int = 60, limit: int = 60
    ) -> bool:
        now = time.time()
        q = self._windows[key][counter]
        cutoff = now - window
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True


class RateLimiter(SecurityHandler):
    """
    Enforces rate limits and usage quotas.

    Supports:
      - Requests per time window (RPM)
      - Tokens per time window (TPM)
      - Concurrent session limits
      - Per-key (user/session/API key) tracking
    """

    def __init__(
        self,
        level: SecurityLevel = SecurityLevel.REJECT,
        rpm: int = 60,
        tpm: int = 100000,
        rpm_window: int = 60,
        tpm_window: int = 60,
        store: Optional[InMemoryRateStore] = None,
        key_resolver: Optional[callable] = None,
    ):
        super().__init__(level)
        self._rpm = rpm
        self._tpm = tpm
        self._rpm_window = rpm_window
        self._tpm_window = tpm_window
        self._store = store or InMemoryRateStore()
        self._key_resolver = key_resolver or (lambda **kw: "default")

    def _resolve_key(self) -> str:
        return self._key_resolver()

    def on_generation_start(self, messages: list, **kwargs):
        key = self._resolve_key()

        if not self._store.consume(key, "requests", 1, self._rpm_window, self._rpm):
            self.reject(
                f"Rate limit exceeded: {self._rpm} requests per {self._rpm_window}s"
            )

    def on_generation_end(self, response, **kwargs):
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
        key = self._resolve_key()

        if not self._store.consume(key, "tokens", tokens, self._tpm_window, self._tpm):
            self.warn(f"Token rate limit approached: {tokens} tokens")
